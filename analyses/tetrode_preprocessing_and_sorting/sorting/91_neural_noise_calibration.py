"""E1 -- neural/noise calibration for the Part C MUA sieve (RESIDUAL_CAPTURE_PLAN s8 E1).

The MUA sieve keeps an unclaimed event if its best cosine to ANY same-tetrode SUA template is >= theta
(neural shape) and below the SUA bar (0.8); events below theta are noise. This calibrates theta and checks
whether BombCell's NP-tuned shape defaults transfer to tetrodes, on two reference sets in a drift-stable
window:
  * NEURAL  = spikes of well-isolated SUA units (isolation gate) -> should read high best-cos-to-bank.
  * NOISE   = random non-peak snippets (>= 1 ms from every detected peak), assigned to random tetrodes
              -> should read low best-cos-to-bank.
theta is picked at a low noise false-positive rate; we report the neural false-negative cost there. Per
the threshold-derivation policy: narrow derivation (this window), broad application (flagged provisional).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/91_neural_noise_calibration.py \
        [--bank assembled_mergefirst] [--window-h 26] [--n-ref 20000]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si

from _assignment_eval import best_template_for_events, window_bank
from _mp_common import FS, materialize_span
from _track_eval import isolation_tier_mask
from _wobble_eval import detect_window_peaks, quality_df

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
SUA_TAU = 0.8  # the SUA acceptance bar; theta must sit below this (MUA = neural-shaped but sub-SUA)


def _pcts(x, ps):
    x = x[np.isfinite(x)]
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} if x.size else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="assembled_mergefirst")
    ap.add_argument("--window-h", type=float, default=26.0)
    ap.add_argument("--n-ref", type=int, default=20000, help="max spikes per reference set")
    ap.add_argument("--tier", default="conservative")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rec = materialize_span(OUT, START_S, DUR_S)
    bank_path = OUT / args.bank
    if not bank_path.exists():
        print(f"bank {args.bank} missing; fall back to assembled_reseed_c12", flush=True)
        bank_path = OUT / "assembled_reseed_c12"
    sorting = si.load(bank_path)
    a = int(args.window_h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win, bank, trains = window_bank(rec, sorting, a, b, n_jobs=N_JOBS)
    if bank is None:
        print("no unit >= template floor in window -- abort", flush=True)
        return
    bank_ids = [int(u) for u in bank.unit_ids]

    # window sorting (local frames) for the isolation gate
    wsort = si.NumpySorting.from_unit_dict([{u: trains[u] for u in bank_ids}], sampling_frequency=FS)
    grp_of = {int(u): int(g) for u, g in zip(sorting.unit_ids, np.asarray(sorting.get_property("group")))}
    wsort.set_property("group", np.array([grp_of[u] for u in bank_ids]))
    qm = quality_df(wsort, win, n_jobs=N_JOBS)
    good = set(int(u) for u in np.asarray(qm.index)[isolation_tier_mask(qm, args.tier)])
    print(f"bank {bank_path.name}: {len(bank_ids)} units, {len(good)} well-isolated ({args.tier})", flush=True)

    # NEURAL reference: spikes of well-isolated units
    n_samp, n_grp = [], []
    for u in good:
        n_samp.append(trains[u])
        n_grp.append(np.full(trains[u].size, grp_of[u], np.int64))
    if not n_samp:
        print("no well-isolated units -- abort", flush=True)
        return
    n_samp = np.concatenate(n_samp)
    n_grp = np.concatenate(n_grp)
    if n_samp.size > args.n_ref:
        idx = rng.choice(n_samp.size, args.n_ref, replace=False)
        n_samp, n_grp = n_samp[idx], n_grp[idx]
    neural_cos, _, neural_valid = best_template_for_events(win, bank, n_samp, n_grp)

    # NOISE reference: random frames >= 1 ms from every detected peak, on random tetrodes
    peak_s, _, _ = detect_window_peaks(win, n_jobs=N_JOBS)
    peak_s = np.sort(peak_s)
    nfr = win.get_num_frames()
    refr = int(1e-3 * FS)
    groups_present = sorted(set(grp_of[u] for u in bank_ids))
    cand = rng.integers(100, nfr - 100, size=args.n_ref * 3)
    j = np.searchsorted(peak_s, cand)
    dprev = np.where(j > 0, cand - peak_s[np.clip(j - 1, 0, peak_s.size - 1)], refr + 1)
    dnext = np.where(j < peak_s.size, peak_s[np.clip(j, 0, peak_s.size - 1)] - cand, refr + 1)
    far = cand[np.minimum(dprev, dnext) > refr][:args.n_ref]
    noise_grp = rng.choice(groups_present, size=far.size)
    noise_cos, _, noise_valid = best_template_for_events(win, bank, far, noise_grp.astype(np.int64))

    nc = neural_cos[neural_valid]
    zc = noise_cos[noise_valid]
    print(f"\nNEURAL best-cos (n={nc.size:,}): " + ", ".join(f"{k}={v:.3f}" for k, v in
          _pcts(nc, [5, 10, 25, 50]).items()))
    print(f"NOISE  best-cos (n={zc.size:,}): " + ", ".join(f"{k}={v:.3f}" for k, v in
          _pcts(zc, [50, 90, 95, 99]).items()))
    # theta candidates: noise FP rate + neural FN rate
    print(f"\n{'theta':>6} {'noise FP%':>10} {'neural FN%':>11}  (MUA band = [theta, {SUA_TAU}))")
    rows = []
    for theta in np.round(np.arange(0.40, 0.81, 0.05), 2):
        fp = float((zc >= theta).mean() * 100) if zc.size else float("nan")
        fn = float((nc < theta).mean() * 100) if nc.size else float("nan")
        rows.append((float(theta), fp, fn))
        print(f"{theta:>6.2f} {fp:>10.2f} {fn:>11.2f}", flush=True)
    # suggested theta = lowest with noise FP <= 5%
    ok = [t for t, fp, _ in rows if np.isfinite(fp) and fp <= 5.0]
    theta_star = min(ok) if ok else float("nan")
    print(f"\nsuggested theta = {theta_star} (lowest with noise FP<=5%); provisional -- recalibrate per bank",
          flush=True)

    # BombCell default-transfer check (best-effort secondary signal)
    bomb = {}
    try:
        from spikeinterface.curation import bombcell_get_default_thresholds, bombcell_label_units
        az = si.create_sorting_analyzer(wsort, win, format="memory",
                                        sparsity=bank.sparsity, return_in_uV=False)
        az.compute(["random_spikes", "noise_levels", "waveforms", "templates", "spike_amplitudes",
                    "template_metrics"], n_jobs=N_JOBS)
        labels = bombcell_label_units(az, bombcell_get_default_thresholds())
        vals, cnts = np.unique(np.asarray(labels), return_counts=True)
        bomb = {str(k): int(v) for k, v in zip(vals, cnts)}
        good_noise = sum(v for k, v in bomb.items() if k == "noise")
        print(f"\nBombCell labels on {len(bank_ids)} SUA units: {bomb} "
              f"({good_noise} labelled noise -- should be LOW if defaults transfer)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\nBombCell check skipped: {type(e).__name__}: {e}", flush=True)

    (OUT / "neural_noise_calibration.json").write_text(json.dumps({
        "bank": bank_path.name, "window_h": args.window_h, "tier": args.tier,
        "n_good_units": len(good), "neural_cos_pcts": _pcts(nc, [5, 10, 25, 50]),
        "noise_cos_pcts": _pcts(zc, [50, 90, 95, 99]),
        "theta_table": [{"theta": t, "noise_fp_pct": fp, "neural_fn_pct": fn} for t, fp, fn in rows],
        "theta_suggested": theta_star, "sua_tau": SUA_TAU, "bombcell_labels": bomb,
    }, indent=2))
    print(f"\nwrote {OUT / 'neural_noise_calibration.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

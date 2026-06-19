"""Pick circus-omp's amplitude gate on its OWN merits (no wobble reference): the refractory knee.

Circus analog of 73_wobble_threshold_intrinsic.py. circus-omp's primary knob is `amplitudes=[lo, hi]`, the
fitted-amplitude acceptance band -- a SCALE-INVARIANT ratio (a = fit/template), unlike wobble's absolute
objective threshold, so one value should generalize across windows. We sweep the lower bound `lo` (hi=inf)
and, for each, score the SAME intrinsic signals as the wobble knee:
  * median rp_contamination (vs BombCell's good bound 0.1) + per-unit rp<0.1 count + isolation tiers,
  * >=12 MAD coverage + amplitude_cutoff (recall) + spurious fraction,
  * MARGINAL-spike quality: of the spikes newly admitted when LOOSENING (lower lo), what fraction land on
    a real >=5.5 MAD peak (real events recovered vs junk)?
Also runs `amplitudes=None` -- circus's BUILT-IN per-template auto lower bound (set from each template's
max_similarity distribution; circus.py:297-302) = a native per-unit gate, reported alongside the global sweep.

The other circus params are left non-binding: `rank=5` auto-clamps to min(rank, n_channels)=4 on tetrodes
(no fix needed, unlike wobble's approx_rank); `omp_min_sps=0.1` is permissive so `amplitudes` is the gate.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes \
        python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/75_circus_amplitude_knee.py [26]
"""
import json
import pathlib
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _mp_common import (_unit_groups_from_mask, build_templates_object, dedup_sorting,
                        materialize_span, run_matching)
from _track_eval import TIERS, isolation_tier_mask
from _wobble_eval import (_by_tet, _within_tol, coverage_by_band, detect_window_peaks,
                          precision_summary, quality_df, spurious_fraction)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
TOL = int(0.5e-3 * FS)
BOMBCELL_RP_MAX = 0.1
AMP_LOWS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # lower bound of amplitudes=[lo, inf]; higher lo = stricter


def score_one(label, method_kwargs, win, bank, ug_map, peak_s, peak_g, amp_mad):
    t0 = time.perf_counter()
    mp, spikes = run_matching(win, bank, method="circus-omp", method_kwargs=method_kwargs, n_jobs=N_JOBS)
    mp.set_property("group", np.array([ug_map[int(u)] for u in mp.unit_ids]))
    dd = dedup_sorting(mp, win)
    qm = quality_df(dd, win, n_jobs=N_JOBS, with_amplitude_cutoff=True)
    prec = precision_summary(dd, qm)
    rp = np.asarray(qm["rp_contamination"], dtype=float)
    ac = np.asarray(qm["amplitude_cutoff"], dtype=float)
    nspk = np.array([dd.get_unit_spike_train(u).size for u in qm.index])
    evalmask = np.isfinite(rp) & (nspk >= 50)
    med_ac = float(np.median(ac[evalmask & np.isfinite(ac)])) if np.any(evalmask & np.isfinite(ac)) else float("nan")
    cov, _ = coverage_by_band(dd, peak_s, peak_g, amp_mad)
    spur = spurious_fraction(dd, peak_s, peak_g)
    tiers = {t: int(isolation_tier_mask(qm, t).sum()) for t in TIERS}
    row = {"label": label, "n_spikes": int(spikes.size), "n_units": int(dd.get_num_units()),
           "median_rp": prec["median_rp_contamination"], "median_amplitude_cutoff": med_ac,
           "rp_good_lt0.1": int(np.sum(evalmask & (rp < BOMBCELL_RP_MAX))), "n_rp_evaluable": int(evalmask.sum()),
           "tier_counts": tiers, "cov12": cov[">=12_pooled"], "spurious": float(spur)}
    print(f"  {label:>12}: {spikes.size:>8,}sp {dd.get_num_units():>3}u | med_rp {row['median_rp']:.4f} | "
          f"amp_cut {med_ac:.3f} | rp<0.1 {row['rp_good_lt0.1']}/{row['n_rp_evaluable']} | tiers "
          f"P{tiers['permissive']}/M{tiers['moderate']}/C{tiers['conservative']} | >=12 {cov['>=12_pooled']:.1f}% | "
          f"spur {spur*100:.1f}% | {time.perf_counter()-t0:.0f}s", flush=True)
    return row, _by_tet(dd)


def main():
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    a = int(h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a, b)
    win.reset_times()
    sdir = WV / "circus_knee" / f"w{int(h)}h_ref"
    shutil.rmtree(sdir, ignore_errors=True)
    t0 = time.perf_counter()
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=True, n_jobs=N_JOBS, seed=0)
    ug_map = {int(u): int(g) for u, g in
              zip(bank.unit_ids, _unit_groups_from_mask(bank.sparsity.mask, rec_groups))}
    peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
    pbt = {int(g): np.sort(peak_s[peak_g == g]) for g in np.unique(peak_g)}
    print(f"window @ {h:.0f}h: {bank.unit_ids.size} units, {peak_s.size:,} events "
          f"({(amp_mad>=12).sum():,} >=12 MAD), setup {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"BombCell good-unit bound: rp_contamination < {BOMBCELL_RP_MAX}\n", flush=True)

    runs, per_tet = [], {}
    for lo in AMP_LOWS:
        row, bt = score_one(f"[{lo:.1f},inf]", {"amplitudes": [lo, np.inf]}, win, bank, ug_map,
                            peak_s, peak_g, amp_mad)
        row["amp_low"] = lo
        runs.append(row)
        per_tet[lo] = bt
    # circus built-in per-template auto lower bound
    auto_row, _ = score_one("auto(None)", {"amplitudes": None}, win, bank, ug_map, peak_s, peak_g, amp_mad)

    # marginal-spike quality: LOOSENING (lo_low < lo_hi) -> are newly-admitted spikes real?
    los = sorted(per_tet)
    for i in range(len(los) - 1):
        lo_loose, lo_strict = los[i], los[i + 1]
        loose, strict = per_tet[lo_loose], per_tet[lo_strict]
        ns, ng = [], []
        for g, st_loose in loose.items():
            st_strict = strict.get(g, np.empty(0, np.int64))
            matched = _within_tol(st_loose, np.full(st_loose.size, g), {g: st_strict}, TOL)
            ns.append(st_loose[~matched])
            ng.append(np.full(int((~matched).sum()), g))
        nsa = np.concatenate(ns) if ns else np.empty(0, np.int64)
        nga = np.concatenate(ng) if ng else np.empty(0, np.int64)
        supp = _within_tol(nsa, nga, pbt, TOL) if nsa.size else np.empty(0, bool)
        marg = float(supp.mean()) if nsa.size else float("nan")
        next(r for r in runs if r["amp_low"] == lo_loose)["marginal_supported_frac"] = marg
        print(f"  loosen {lo_strict:.1f}->{lo_loose:.1f}: +{nsa.size:,} spikes, {marg*100:.0f}% on a >=5.5 MAD peak",
              flush=True)

    asc = sorted(runs, key=lambda r: r["amp_low"])
    bombcell_knee = next((r["amp_low"] for r in reversed(asc)
                          if np.isfinite(r["median_rp"]) and r["median_rp"] < BOMBCELL_RP_MAX), None)
    clean_knee = next((r["amp_low"] for r in reversed(asc)
                       if np.isfinite(r["median_rp"]) and r["median_rp"] < 0.03), None)
    print(f"\nKNEE: lowest amplitudes[0] with median rp<0.1 = {bombcell_knee}; rp<0.03 = {clean_knee}; "
          f"auto(None) med_rp={auto_row['median_rp']:.4f} cov={auto_row['cov12']:.1f}%", flush=True)

    la = np.array([r["amp_low"] for r in asc])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(la, [r["median_rp"] for r in asc], "o-", color="#3b7dd8", label="median rp_contamination")
    ax[0].axhline(BOMBCELL_RP_MAX, color="#c0392b", ls="--", label=f"BombCell good bound ({BOMBCELL_RP_MAX})")
    ax[0].axhline(auto_row["median_rp"], color="#2e8b57", ls=":", label="auto(None) rp")
    ax[0].set_xlabel("amplitudes[0] (lower bound; higher = stricter)")
    ax[0].set_ylabel("median rp_contamination")
    ax[0].set_title(f"circus-omp refractory knee @ {h:.0f}h")
    ax[0].legend(fontsize=8)
    ax[1].plot(la, [r["cov12"] for r in asc], "o-", color="#2e8b57", label=">=12 MAD coverage %")
    ax[1].plot(la, [r["spurious"] * 100 for r in asc], "s--", color="#7d3bd8", label="spurious %")
    mla = [r["amp_low"] for r in asc if "marginal_supported_frac" in r]
    mva = [r["marginal_supported_frac"] * 100 for r in asc if "marginal_supported_frac" in r]
    ax[1].plot(mla, mva, "d:", color="0.4", label="marginal added-spike supported %")
    ax[1].set_xlabel("amplitudes[0] (lower bound)")
    ax[1].set_ylabel("%")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("Coverage / over-detection vs amplitude bound")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figp = WV / f"circus_knee_w{int(h)}h.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)

    out = {"window_h": h, "bombcell_rp_max": BOMBCELL_RP_MAX, "bombcell_knee_amp_low": bombcell_knee,
           "clean_knee_amp_low": clean_knee, "auto_none": auto_row, "runs": runs}
    (WV / f"circus_knee_w{int(h)}h.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / f'circus_knee_w{int(h)}h.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

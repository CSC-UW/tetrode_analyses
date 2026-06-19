"""Phase A: gating-strategy bake-off for wobble-as-primary (extends 74 across windows + arms).

Per window, on >=10 MAD coverage (the task-1 spec) at matched precision + low-amplitude retention:
  (i)   wobble + adaptive-||t||^2 absolute threshold -- sweep factor (threshold = factor x tsq_median).
        Each factor is a SEPARATE wobble run (the threshold changes detection).
  (ii)  wobble + cosine shape gate                   -- ONE permissive admit run (the smallest factor),
        then sweep the scale-invariant per-spike cosine r post-hoc (free; reuses per_spike_cosine).
  (iii) wobble at its intrinsic knee                 -- a marked factor within (i) (script-73 knee ~0.68).
  ref:  circus-omp [0.8, inf]                         -- the scale-invariant amplitude-ratio reference.

GENERALIZATION = spread of (median rp, >=10 cov) ACROSS windows at each fixed factor / r. The cosine
gate's claim is recalibration-free generalization (one r across windows) + low-amp retention; this is
what decides it vs the absolute threshold (whose single factor under-generalised -- scripts 74/75).
GROWN-BANK generalisation (the bank carried/re-seeded across 48 h) is covered by Phase C/D (script 81)
on the reseed deliverable, so it is NOT duplicated here. Scoring is PRE-DEDUP (like 74): the gate's
effect on rp/coverage is the signal; dedup is an orthogonal downstream step applied equally to all arms.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/79_wobble_primary_gate_bakeoff.py \
        [--windows-h 3 11 19 27 35 43] [--abs-factors 0.45 0.55 0.68 0.80] [--r-gates 0.5 0.6 0.7 0.8]
"""
import argparse
import json
import pathlib
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _mp_common import (_unit_groups_from_mask, build_templates_object, materialize_span,
                        per_spike_cosine, run_matching, tsq_median, wobble_method_kwargs)
from _wobble_eval import detect_window_peaks, score_kept_spikes
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", type=float, nargs="+", default=[3.0, 11.0, 19.0, 27.0, 35.0, 43.0],
                    help="window start times in hours into the materialized span (0 .. ~47 h)")
    ap.add_argument("--abs-factors", type=float, nargs="+", default=[0.45, 0.55, 0.68, 0.80],
                    help="adaptive-||t||^2 thresholds (x tsq_median); the smallest doubles as the cosine "
                    "arm's permissive admit. 0.55=circus-matched, 0.68=intrinsic knee (script 73).")
    ap.add_argument("--r-gates", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8],
                    help="cosine shape-gate values (scale-invariant; post-filtered on the permissive run)")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for output gate_bakeoff{tag}.{json,png} (avoid clobber; "
                    "e.g. _finer_r for a fine cosine-only sweep with --abs-factors 0.45)")
    args = ap.parse_args()
    abs_factors = sorted(args.abs_factors)
    permissive = abs_factors[0]
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    results = {}
    for h in args.windows_h:
        a0 = int(h * 3600 * FS)
        b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a0, b0)
        win.reset_times()
        nfr = win.get_num_frames()
        sdir = WV / "bakeoff" / f"w{int(h)}h_ref"
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.perf_counter()
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=args.n_jobs, seed=0)
        ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
        med = tsq_median(bank)
        peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=args.n_jobs)
        peak_by_tet = {int(gg): np.sort(peak_s[peak_g == gg]) for gg in np.unique(peak_g)}
        print(f"\n=== @ {h:.0f}h: {bank.unit_ids.size} units, {peak_s.size:,} events "
              f"({(amp_mad >= 10).sum():,} >=10 MAD), tsq_med={med:.3g}, setup {time.perf_counter()-t0:.0f}s ===",
              flush=True)

        def sc(s, ci):
            return score_kept_spikes(s, ci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)

        _, cs = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=args.n_jobs)
        circ = sc(cs["sample_index"].astype(np.int64), cs["cluster_index"].astype(np.int64))
        print(f"  circus[0.8,inf]: rp {circ['median_rp']:.4f} | >=10 {circ['cov10']:.1f}% | "
              f"low {circ['cov_low']:.1f}% | spur {circ['spurious']*100:.0f}% | {circ['n_units']}u", flush=True)

        abs_rows, perm_spikes = {}, None
        for f in abs_factors:
            tw = time.perf_counter()
            _, sp = run_matching(win, bank, method="wobble", n_jobs=args.n_jobs,
                                 method_kwargs=wobble_method_kwargs(bank, threshold=f * med))
            row = sc(sp["sample_index"].astype(np.int64), sp["cluster_index"].astype(np.int64))
            abs_rows[f] = row
            if f == permissive:
                perm_spikes = sp
            print(f"  abs {f:.2f}x: rp {row['median_rp']:.4f} | >=10 {row['cov10']:.1f}% | "
                  f"low {row['cov_low']:.1f}% | spur {row['spurious']*100:.0f}% | {row['n_units']}u "
                  f"| {time.perf_counter()-tw:.0f}s", flush=True)

        r_all = per_spike_cosine(perm_spikes, bank, win)
        ps_s = perm_spikes["sample_index"].astype(np.int64)
        ps_ci = perm_spikes["cluster_index"].astype(np.int64)
        cos_rows = {}
        for rg in args.r_gates:
            keep = r_all >= rg
            row = sc(ps_s[keep], ps_ci[keep])
            cos_rows[rg] = row
            print(f"  cos r>={rg:.2f}: rp {row['median_rp']:.4f} | >=10 {row['cov10']:.1f}% | "
                  f"low {row['cov_low']:.1f}% | spur {row['spurious']*100:.0f}% | {row['n_units']}u "
                  f"(keep {keep.mean()*100:.0f}%)", flush=True)

        results[f"{h:.0f}h"] = {"tsq_median": med, "circus": circ,
                                "absolute": {f"{f:.2f}": abs_rows[f] for f in abs_factors},
                                "cosine": {f"{rg:.2f}": cos_rows[rg] for rg in args.r_gates}}

    _summarize_and_plot(results, abs_factors, args.r_gates, args.tag)


def _spread(results, arm, key, field):
    """(min, max, max-min) of `field` across windows at fixed gate `key` within `arm`."""
    vals = [results[h][arm][key][field] for h in results]
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(v.min()), float(v.max()), float(v.max() - v.min())) if v.size else (np.nan, np.nan, np.nan)


def _summarize_and_plot(results, abs_factors, r_gates, tag=""):
    hs = list(results)
    print("\n=== generalization across windows (spread = max-min at a fixed gate) ===", flush=True)
    print("  circus[0.8,inf]: rp range "
          f"{min(results[h]['circus']['median_rp'] for h in hs):.3f}-"
          f"{max(results[h]['circus']['median_rp'] for h in hs):.3f} | >=10 cov range "
          f"{min(results[h]['circus']['cov10'] for h in hs):.1f}-"
          f"{max(results[h]['circus']['cov10'] for h in hs):.1f}", flush=True)
    for arm, gates in (("absolute", [f"{f:.2f}" for f in abs_factors]),
                       ("cosine", [f"{rg:.2f}" for rg in r_gates])):
        for k in gates:
            rp = _spread(results, arm, k, "median_rp")
            cov = _spread(results, arm, k, "cov10")
            low = _spread(results, arm, k, "cov_low")
            print(f"  {arm[:3]} {k}: rp {rp[0]:.3f}-{rp[1]:.3f} (spread {rp[2]:.3f}) | "
                  f">=10 cov {cov[0]:.1f}-{cov[1]:.1f} (spread {cov[2]:.1f}) | "
                  f"low-amp {low[0]:.1f}-{low[1]:.1f}", flush=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(hs)))
    for col, (arm, gates, xlabel) in enumerate(
            (("absolute", abs_factors, "factor x tsq_median (threshold)"),
             ("cosine", r_gates, "cosine gate r"))):
        gk = [f"{g:.2f}" for g in gates]
        for hi, h in enumerate(hs):
            ax[0][col].plot(gates, [results[h][arm][k]["cov10"] for k in gk], "o-", color=colors[hi], label=h)
            ax[1][col].plot(gates, [results[h][arm][k]["median_rp"] for k in gk], "o-", color=colors[hi])
        cc = [results[h]["circus"] for h in hs]
        ax[0][col].axhline(np.mean([c["cov10"] for c in cc]), color="0.4", ls="--", label="circus mean")
        ax[1][col].axhline(np.mean([c["median_rp"] for c in cc]), color="0.4", ls="--", label="circus mean")
        ax[1][col].axhline(0.1, color="#c0392b", ls=":", label="BombCell 0.1")
        ax[0][col].set_title(f"{arm} arm: >=10 MAD coverage")
        ax[0][col].set_ylabel("% >=10 MAD claimed")
        ax[0][col].set_ylim(0, 102)
        ax[0][col].legend(fontsize=7, ncol=2)
        ax[1][col].set_title(f"{arm} arm: median rp_contamination")
        ax[1][col].set_ylabel("median rp")
        ax[1][col].set_xlabel(xlabel)
        ax[1][col].legend(fontsize=7)
    fig.suptitle("Phase A gate bake-off: does ONE fixed gate value generalize across windows?\n"
                 "(tight line clustering = generalizes; cosine arm should also hold low-amp coverage)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figp = WV / f"gate_bakeoff{tag}.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)
    jsonp = WV / f"gate_bakeoff{tag}.json"
    jsonp.write_text(json.dumps({"abs_factors": abs_factors, "r_gates": r_gates, "results": results}, indent=2))
    print(f"\nwrote {figp}\nwrote {jsonp}\nDONE", flush=True)


if __name__ == "__main__":
    main()

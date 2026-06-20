"""Unified three-axis (A + B + C) head-to-head: MP variants vs the chunk+match deliverable.

"Let the scoreboard decide" the production baseline (the user's choice): score every candidate on the
three axes the goal actually has -- (A) event coverage, (B) per-unit assignment purity, (C) identity
stability -- so the chunk+match (2204 oversplit units, ~99.5% coverage) vs MP (~95-109 units) trade-off
is SURFACED, not assumed. The "wobble Pareto-dominates" headline used only A (pooled) + median rp; this
adds the missing per-unit axis B.

Each variant is scored on its OWN recording (MP variants in the MP materialized binary; chunk+match in
its tracked_48h recording). Axis A: MP variants -> coverage_by_band on the shared MP peaks
(spike_coverage.npz); chunk+match -> band means of its precomputed claimed mask
(spike_coverage_chunkmatch.npz, same peak set). Axis B/C: windowed cosine purity + CCG (script-86 method).
chunk+match axis C (2204 units) is SKIPPED by default (O(units/tetrode^2) CCG); --chunkmatch-ccg forces it.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/89_scoreboard.py [--windows-h 5 26 40]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si

from _mp_common import materialize_span
from _scoreboard import (axis_b_aggregate, axis_c_summary, compare_scoreboards, coverage_by_band,
                        windowed_axis_b)

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix")
OUT = ROOT / "track_eval/mp_long_s2000_d170000"
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
MP_VARIANTS = {"reestimate": "assembled_reestimate", "reseed_c12": "assembled_reseed_c12",
               "dedup095": "assembled_reestimate_dedup095", "merge-first": "assembled_mergefirst"}
CHUNKMATCH = ROOT / "tracked_48h/analyzer_clustered.zarr"


def coverage_from_claimed(claimed, amp_mad):
    """Axis-A coverage dict ({'>=10_pooled','>=12_pooled','overall'}) from a precomputed claimed mask."""
    res = {}
    for lo in (10, 12):
        big = amp_mad >= lo
        res[f">={lo}_pooled"] = float(claimed[big].mean() * 100) if big.any() else float("nan")
    res["overall"] = float(claimed.mean() * 100)
    return res


def score(label, sorting, recording, axis_a, windows, *, axis_c_pairs=None, max_units_full_ccg=400):
    cp_full, cp_tight = windowed_axis_b(recording, sorting, windows, win_s=WIN_S, n_jobs=N_JOBS)
    axis_b = axis_b_aggregate(cp_full, sorting, win_s=WIN_S)
    sw_t = ([c["best_match_frac"] for c in cp_tight.values() if np.isfinite(c["best_match_frac"])])
    num = sum(c["n_finite"] * c["best_match_frac"] for c in cp_tight.values()
              if np.isfinite(c["best_match_frac"]))
    den = sum(c["n_finite"] for c in cp_tight.values() if np.isfinite(c["best_match_frac"]))
    axis_b["spikeweighted_purity_tight"] = float(num / den) if den else float("nan")
    axis_b["median_best_match_frac_tight"] = float(np.median(sw_t)) if sw_t else float("nan")
    axis_c = axis_c_summary(sorting, win_s=WIN_S, pairs=axis_c_pairs,
                            max_units_for_full_ccg=max_units_full_ccg)
    return {"label": label, "n_units": int(sorting.get_num_units()),
            "axis_A": axis_a, "axis_B": axis_b, "axis_C": axis_c}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--chunkmatch-ccg", action="store_true", help="force full axis-C CCG on chunk+match (2204u, slow)")
    args = ap.parse_args()

    mp_rec = materialize_span(OUT, START_S, DUR_S)
    z = np.load(OUT / "spike_coverage.npz", mmap_mode="r")
    peak_s, peak_g, amp_mad = np.asarray(z["peak_sample"]), np.asarray(z["peak_group"]), np.asarray(z["amp_mad"])

    results = {}
    for label, name in MP_VARIANTS.items():
        print(f"\n=== {label} ({name}) [MP binary] ===", flush=True)
        srt = si.load(OUT / name)
        axis_a, _ = coverage_by_band(srt, peak_s, peak_g, amp_mad)
        results[label] = score(label, srt, mp_rec, axis_a, args.windows_h)
        _report(results[label])

    if CHUNKMATCH.exists():
        print("\n=== chunk+match (tracked_48h/analyzer_clustered.zarr) [own recording] ===", flush=True)
        azc = si.load_sorting_analyzer(CHUNKMATCH, load_extensions=False)
        cm_claimed = np.asarray(np.load(OUT / "spike_coverage_chunkmatch.npz", mmap_mode="r")["claimed"])
        axis_a = coverage_from_claimed(cm_claimed, amp_mad)
        results["chunk+match"] = score("chunk+match", azc.sorting, azc.recording, axis_a, args.windows_h,
                                       max_units_full_ccg=10**9 if args.chunkmatch_ccg else 400)
        _report(results["chunk+match"])
    else:
        print(f"\n(chunk+match deliverable missing: {CHUNKMATCH})", flush=True)

    print("\n" + "=" * 90)
    print("UNIFIED THREE-AXIS SCOREBOARD (A coverage / B per-unit purity / C identity stability)")
    print("=" * 90)
    print(compare_scoreboards(results))
    print("\nReading: high covA + low purB_t / many C_dup = coverage bought with oversplit + cross-unit "
          "leakage; the production choice trades these off explicitly, it is not 'whoever covers most'.")

    drop = ("per_unit", "cosine_purity", "tight_purity")
    summ = {lab: {"n_units": r["n_units"], "axis_A": r["axis_A"],
                  "axis_B": {k: v for k, v in r["axis_B"].items() if k not in drop}, "axis_C": r["axis_C"]}
            for lab, r in results.items()}
    (OUT / "scoreboard_unified.json").write_text(json.dumps(summ, indent=2))
    print(f"\nwrote {OUT / 'scoreboard_unified.json'}\nDONE", flush=True)


def _report(r):
    a, b, c = r["axis_A"], r["axis_B"], r["axis_C"]
    print(f"  axis A: >=10 {a.get('>=10_pooled', float('nan')):.1f}%  >=12 {a.get('>=12_pooled', float('nan')):.1f}%",
          flush=True)
    print(f"  axis B: purity full {b['spikeweighted_purity']:.3f} / tight {b['spikeweighted_purity_tight']:.3f}; "
          f"{b['n_flagged']}/{b['n_units']} flagged ({b['n_cross_contaminated']} cross-contam)", flush=True)
    print(f"  axis C: {c['n_units']} units, {c.get('n_ccg_duplicate_pairs')} CCG-dup pairs"
          f"{' (CCG skipped)' if c.get('ccg_skipped') else ''}, max {c['max_units_per_tetrode']}/tet", flush=True)


if __name__ == "__main__":
    main()

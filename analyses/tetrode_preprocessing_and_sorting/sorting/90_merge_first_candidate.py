"""Build the MERGE-FIRST base: CCG-guarded within-tetrode merge of a reseed sorting, then re-score B/C.

The 2026-06-19 re-score showed the dominant defect is OVERSPLIT (110 CCG-duplicate same-tetrode pairs in
the 109-unit base), and competitive reassignment on the oversplit sorting is mostly churn (script 88). So
the production order is MERGE first. This builds that merged base from a reseed sorting via
``ccg_guarded_merge`` (cosine proposes >= 0.90; CCG-duplicate accepts; cosine >= 0.95 fallback for
CCG-abstaining low-co-activity pairs), persists it as ``assembled_mergefirst``, and re-scores axes B (full
+ tight purity) and C (CCG-duplicate pairs) at the representative windows to confirm merge improves both.
Competitive reassignment + residual SUA capture + the per-tetrode MUA bucket (Part C) build ON TOP of this
base (scripts 91/92).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/90_merge_first_candidate.py \
        [--base assembled_reseed_c12] [--propose-cos 0.90] [--fallback-cos 0.95] [--windows-h 5 26 40]
"""
import argparse
import json
import pathlib
from collections import Counter

import numpy as np
import spikeinterface as si

from _assignment_eval import ccg_guarded_merge
from _mp_common import materialize_span
from _scoreboard import axis_b_aggregate, axis_c_summary, windowed_axis_b

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16


def bc_summary(rec, sorting, windows):
    cp_full, cp_tight = windowed_axis_b(rec, sorting, windows, win_s=WIN_S, n_jobs=N_JOBS)
    b = axis_b_aggregate(cp_full, sorting, win_s=WIN_S)
    num = sum(c["n_finite"] * c["best_match_frac"] for c in cp_tight.values()
              if np.isfinite(c["best_match_frac"]))
    den = sum(c["n_finite"] for c in cp_tight.values() if np.isfinite(c["best_match_frac"]))
    b["spikeweighted_purity_tight"] = float(num / den) if den else float("nan")
    c = axis_c_summary(sorting, win_s=WIN_S)
    return b, c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="assembled_reseed_c12")
    ap.add_argument("--propose-cos", type=float, default=0.90)
    ap.add_argument("--fallback-cos", type=float, default=0.95)
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--out-name", default="assembled_mergefirst")
    args = ap.parse_args()

    rec = materialize_span(OUT, START_S, DUR_S)
    base = si.load(OUT / args.base)
    print(f"merge-first base: {args.base} ({base.get_num_units()} units); propose>={args.propose_cos}, "
          f"fallback>={args.fallback_cos}", flush=True)

    merged, merges = ccg_guarded_merge(base, rec, propose_cos=args.propose_cos,
                                       fallback_cos=args.fallback_cos, win_s=WIN_S, n_jobs=N_JOBS)
    by_reason = Counter("duplicate(CCG)" if v == "duplicate" else f"fallback-cos({v})"
                        for _, _, v, _ in merges)
    print(f"  merged {base.get_num_units()} -> {merged.get_num_units()} units; {len(merges)} pair-merges "
          f"accepted: " + ", ".join(f"{k}={n}" for k, n in by_reason.items()), flush=True)
    out_dir = OUT / args.out_name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    merged.save(folder=out_dir)
    print(f"  saved -> {out_dir}", flush=True)

    print("\nre-score (base vs merge-first), windows", args.windows_h, flush=True)
    bb, bc = bc_summary(rec, base, args.windows_h)
    mb, mc = bc_summary(rec, merged, args.windows_h)
    print(f"  {'':12} {'units':>6} {'purB(full)':>11} {'purB(tight)':>12} {'CCG-dup pairs':>14} {'max/tet':>8}")
    for name, b, c, n in [("base", bb, bc, base.get_num_units()),
                          ("merge-first", mb, mc, merged.get_num_units())]:
        print(f"  {name:12} {n:>6} {b['spikeweighted_purity']:>11.3f} {b['spikeweighted_purity_tight']:>12.3f} "
              f"{c.get('n_ccg_duplicate_pairs'):>14} {c['max_units_per_tetrode']:>8}", flush=True)

    (OUT / "merge_first_candidate.json").write_text(json.dumps({
        "base": args.base, "n_base": base.get_num_units(), "n_merged": merged.get_num_units(),
        "n_merges": len(merges), "merge_reasons": dict(by_reason),
        "merges": [{"a": a, "b": b, "verdict": v, "cosine": round(cs, 3)} for a, b, v, cs in merges],
        "base_axisB": {k: bb[k] for k in ("spikeweighted_purity", "spikeweighted_purity_tight")},
        "base_axisC_ccg_dup": bc.get("n_ccg_duplicate_pairs"),
        "merged_axisB": {k: mb[k] for k in ("spikeweighted_purity", "spikeweighted_purity_tight")},
        "merged_axisC_ccg_dup": mc.get("n_ccg_duplicate_pairs"),
    }, indent=2))
    print(f"\nwrote {OUT / 'merge_first_candidate.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

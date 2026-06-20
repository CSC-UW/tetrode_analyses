"""Sweep the per-window TEMPLATE-RELIABILITY floor (min_spikes_template) and check the variant RANKING is
stable. The axis-B scorer builds a window-local template per unit and scores every spike's cosine against
it; a too-low floor means noisy templates (the default was wrongly inherited from _wobble_eval.MIN_SPK=50,
the rp-ESTIMABILITY floor, not _mp_common's template-reliability floor of 100). This measures, at floors
{50,100,200} over the representative windows, (a) how many units each variant retains and (b) whether the
spike-weighted full/tight purity and the across-variant ranking move -- the de-risk for locking the
production floor.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/93_template_floor_sweep.py \
        [--windows-h 5 26 40] [--floors 50 100 200]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si

from _mp_common import materialize_span
from _scoreboard import windowed_axis_b

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
VARIANTS = {"reestimate": "assembled_reestimate", "reseed_c12": "assembled_reseed_c12",
            "dedup095": "assembled_reestimate_dedup095", "merge-first": "assembled_mergefirst"}


def sw(cp):
    """(spike-weighted purity, n units scored) from a cosine_purity dict."""
    fin = [c for c in cp.values() if np.isfinite(c["best_match_frac"])]
    num = sum(c["n_finite"] * c["best_match_frac"] for c in fin)
    den = sum(c["n_finite"] for c in fin)
    return (float(num / den) if den else float("nan")), len(fin)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--floors", nargs="+", type=int, default=[50, 100, 200])
    args = ap.parse_args()
    rec = materialize_span(OUT, START_S, DUR_S)

    results = {}  # (variant, floor) -> dict
    for label, name in VARIANTS.items():
        srt = si.load(OUT / name)
        for floor in args.floors:
            cp_full, cp_tight = windowed_axis_b(rec, srt, args.windows_h, win_s=WIN_S, n_jobs=N_JOBS,
                                                min_spikes_template=floor)
            pf, nf = sw(cp_full)
            pt, _ = sw(cp_tight)
            results[(label, floor)] = {"purity_full": pf, "purity_tight": pt, "n_scored": nf}
            print(f"  {label:12} floor={floor:<4} n_scored={nf:<4} purity full={pf:.3f} tight={pt:.3f}",
                  flush=True)

    print("\n" + "=" * 70)
    print("TEMPLATE-FLOOR SWEEP -- tight-purity ranking per floor (is it stable?)")
    print("=" * 70)
    print(f"{'floor':>6} | " + " > ".join(["ranking by tight purity (best first)"]))
    rankings = {}
    for floor in args.floors:
        order = sorted(VARIANTS, key=lambda v: -results[(v, floor)]["purity_tight"])
        rankings[floor] = order
        cells = [f"{v}({results[(v, floor)]['purity_tight']:.3f})" for v in order]
        print(f"{floor:>6} | " + "  >  ".join(cells), flush=True)
    stable = len({tuple(rankings[f]) for f in args.floors}) == 1
    print(f"\nranking floor-STABLE across {args.floors}: {stable}", flush=True)
    # retention cost of raising the floor
    print("\nunits scored (retention) by floor:")
    print(f"  {'variant':12} " + " ".join(f"f{f:>4}" for f in args.floors))
    for v in VARIANTS:
        print(f"  {v:12} " + " ".join(f"{results[(v, f)]['n_scored']:>5}" for f in args.floors), flush=True)

    (OUT / "template_floor_sweep.json").write_text(json.dumps(
        {f"{v}__{f}": results[(v, f)] for v, f in results}
        | {"ranking_stable": bool(stable), "rankings": {str(f): rankings[f] for f in args.floors}},
        indent=2))
    print(f"\nwrote {OUT / 'template_floor_sweep.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""Re-score the on-disk MP assembled sortings on axes B (assignment purity) and C (identity stability).

The wobble / coverage work was judged on tetrode-POOLED coverage + median rp_contamination -- both blind
to PER-UNIT assignment correctness (an independently-firing same-tetrode neighbour mis-assigned into a
unit lands on the right tetrode -> "covered", and violates no refractory -> rp~0; see _assignment_eval +
the reorientation plan). This script answers the review's points #1/#2 directly, with NO 48 h re-sort:
it re-uses the assembled sortings already on disk and the cached materialized binary.

Axis B (per-unit purity): for each variant, walk a set of windows; in each, build window-local templates
(drift-appropriate) and ask, for every window spike, whether its ASSIGNED template is the best
same-tetrode cosine match (all_template_cosines). Accumulate across windows -> per-unit best_match_frac;
adjudicate each unit's top cosine neighbour with the CCG arbiter (DISTINCT = real cross-unit
contamination; DUPLICATE = oversplit). Axis C (cheap, spike-trains only): same-tetrode CCG 'duplicate'
pairs + unit-count parsimony. Held-out agreement (signal 3) is the expensive third signal -- off by
default (pass --heldout to add it).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/86_rescore_axis_b_c.py \
        [--variants assembled_reestimate assembled_reseed_c12 ...] [--windows-h 5 26 40] [--heldout]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si

from _assignment_eval import (accumulate_best_match, finalize_best_match, heldout_window_agreement,
                              window_assignment_cosines)
from _mp_common import FS, materialize_span
from _scoreboard import axis_b_aggregate, axis_c_summary, compare_scoreboards

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16

# Default variants: the MP-frame production CANDIDATES (all live in the MP materialized binary, so they
# share one recording here) -- base carry-forward, the reseed deliverable, the deduped sorting. The 2204-u
# chunk+match deliverable lives in a DIFFERENT recording (tracked_48h) and is scored in script 89, which
# scores each variant on its own recording. (analyzer_tracks.zarr is just the analyzer over
# assembled_reestimate -- same sorting -- so it is NOT a separate variant.)
DEFAULT_VARIANTS = {
    "reestimate": "assembled_reestimate",
    "reseed_c12": "assembled_reseed_c12",
    "dedup095": "assembled_reestimate_dedup095",
}


def load_sorting(name):
    """Load an assembled NumpySorting (folder) or a SortingAnalyzer's sorting (.zarr)."""
    p = OUT / name
    if name.endswith(".zarr"):
        return si.load_sorting_analyzer(p).sorting
    return si.load(p)


def _spikeweighted(cosine_purity):
    num = sum(c["n_finite"] * c["best_match_frac"] for c in cosine_purity.values()
              if np.isfinite(c["best_match_frac"]))
    den = sum(c["n_finite"] for c in cosine_purity.values() if np.isfinite(c["best_match_frac"]))
    fr = [c["best_match_frac"] for c in cosine_purity.values() if np.isfinite(c["best_match_frac"])]
    return (float(num / den) if den else float("nan")), (float(np.median(fr)) if fr else float("nan"))


def score_variant(rec, sorting, windows, *, with_heldout=False):
    """Axis B (windowed cosine purity + CCG) + axis C for one sorting.

    Scores BOTH the full-window cosine (rF) and the tight trough window (rA) -- the full window is
    forgiving (gross shape), the tight trough is where co-tetrode units differ most (script 85's
    54%-neighbour-wins finding was a TIGHT-window result), so the gap between them is itself the signal.
    """
    nfr = rec.get_num_frames()
    acc_full: dict = {}
    acc_tight: dict = {}
    for h in windows:
        a = int(h * 3600 * FS)
        b = min(a + int(WIN_S * FS), nfr)
        if a >= nfr:
            print(f"    window {h}h beyond recording ({nfr / FS / 3600:.1f}h) -- skip", flush=True)
            continue
        spikes, cosines, bank_ids = window_assignment_cosines(rec, sorting, a, b, n_jobs=N_JOBS)
        if spikes is None:
            print(f"    window {h}h: no unit >= template floor -- skip", flush=True)
            continue
        accumulate_best_match(acc_full, spikes, cosines, bank_ids, use_tight=False)
        accumulate_best_match(acc_tight, spikes, cosines, bank_ids, use_tight=True)
        print(f"    window {h}h: {spikes.size:,} spikes scored over {len(bank_ids)} units", flush=True)
    cosine_purity = finalize_best_match(acc_full)
    tight_purity = finalize_best_match(acc_tight)
    heldout = heldout_window_agreement(rec, sorting, win_s=WIN_S, n_jobs=N_JOBS) if with_heldout else None
    axis_b = axis_b_aggregate(cosine_purity, sorting, win_s=WIN_S, heldout=heldout)
    sw_t, med_t = _spikeweighted(tight_purity)
    axis_b["spikeweighted_purity_tight"] = sw_t
    axis_b["median_best_match_frac_tight"] = med_t
    axis_b["tight_purity"] = tight_purity
    axis_c = axis_c_summary(sorting, win_s=WIN_S)
    return {"n_units": int(sorting.get_num_units()), "axis_B": axis_b, "axis_C": axis_c}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="variant names (assembled_* dir or *.zarr); default = the 4 candidates")
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--heldout", action="store_true", help="also compute signal-3 held-out agreement (slow)")
    args = ap.parse_args()
    variants = ({v: v for v in args.variants} if args.variants else DEFAULT_VARIANTS)

    rec = materialize_span(OUT, START_S, DUR_S)
    print(f"recording {rec.get_num_frames() / FS / 3600:.1f}h; windows {args.windows_h}h; "
          f"heldout={args.heldout}", flush=True)
    results = {}
    for label, name in variants.items():
        print(f"\n=== {label} ({name}) ===", flush=True)
        sorting = load_sorting(name)
        res = score_variant(rec, sorting, args.windows_h, with_heldout=args.heldout)
        res["label"] = label
        b, c = res["axis_B"], res["axis_C"]
        print(f"  axis B: spike-weighted purity full {b['spikeweighted_purity']:.3f} / tight "
              f"{b['spikeweighted_purity_tight']:.3f} (median full {b['median_best_match_frac']:.3f}), "
              f"{b['n_flagged']}/{b['n_units']} flagged ({b['n_cross_contaminated']} cross-contam, "
              f"{b['n_oversplit']} oversplit)", flush=True)
        print(f"  axis C: {c['n_units']} units, {c.get('n_ccg_duplicate_pairs')} CCG-duplicate same-tet pairs, "
              f"max {c['max_units_per_tetrode']}/tetrode", flush=True)
        results[label] = res

    print("\n" + "=" * 80)
    print("THREE-AXIS RE-SCORE (axes B+C; coverage axis A is script 89)")
    print("=" * 80)
    print(compare_scoreboards(results))

    # persist: compact JSON summary (no big per-unit dicts) + per-unit purity arrays per variant
    _drop = ("per_unit", "cosine_purity", "tight_purity")
    summ = {label: {"n_units": r["n_units"],
                    "axis_B": {k: v for k, v in r["axis_B"].items() if k not in _drop},
                    "axis_C": r["axis_C"]}
            for label, r in results.items()}
    (OUT / "axis_bc_rescore.json").write_text(json.dumps(summ, indent=2))
    npz = {}
    for label, r in results.items():
        cp = r["axis_B"]["cosine_purity"]
        uids = np.array(sorted(cp))
        npz[f"{label}__uid"] = uids
        npz[f"{label}__best_match_frac"] = np.array([cp[u]["best_match_frac"] for u in uids])
        tp = r["axis_B"]["tight_purity"]
        npz[f"{label}__best_match_frac_tight"] = np.array([tp.get(u, {}).get("best_match_frac", np.nan)
                                                           for u in uids])
        npz[f"{label}__top_neighbor"] = np.array([cp[u]["top_neighbor"] for u in uids])
        npz[f"{label}__n_finite"] = np.array([cp[u]["n_finite"] for u in uids])
        npz[f"{label}__category"] = np.array([r["axis_B"]["per_unit"][u]["category"] for u in uids])
    np.savez(OUT / "axis_bc_rescore.npz", **npz)
    print(f"\nwrote {OUT / 'axis_bc_rescore.json'} and .npz\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""Competitive assignment vs the delete-only shape gate (measure-first, post-hoc).

run_matching's ``shape_gate_r`` DELETES a low-cosine spike; ``competitive_reassign`` instead OFFERS it to
every same-tetrode template and re-labels it to the best match (reusing all_template_cosines). This
measures, on the on-disk MP deliverable, how many spikes post-hoc competitive reassignment MOVES and --
crucially -- adjudicates each move with the CCG arbiter: a (from -> to) pair that is a DISTINCT cell
(filled CCG) means the move fixed a real cross-unit mis-assignment; a DUPLICATE pair (refractory dip)
means it just shuffled spikes between two tracks of the same over-split cell (harmless churn, an axis-C
merge issue). Reported for both the full window (rF, gross shape) and the tight trough window (rA, where
co-tetrode units differ most -- script 85's neighbour-win regime).

Post-hoc re-labeling MEASUREMENT only -- NOT a change to the matcher. Folding competition into the
carry-forward loops is justified only if moves are predominantly to DISTINCT neighbours (real fixes).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/88_competitive_reassign.py \
        [--variant assembled_reestimate] [--windows-h 5 26 40] [--margin 0.0] [--min-pair 20]
"""
import argparse
import json
import pathlib
from collections import Counter

import numpy as np
import spikeinterface as si

from _assignment_eval import ccg_verdict_pair, window_assignment_cosines
from _mp_common import FS, competitive_reassign, materialize_span

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16


def classify_pairs(pair_counts, sorting, win_frames, min_pair):
    """For each (from_uid -> to_uid) reassignment pair, the CCG verdict on the two FULL trains. Returns
    (verdict -> reassigned-spike mass, sorted detail rows). distinct = real cross-unit fix; duplicate =
    over-split churn; too_few = below the CCG co-activity floor."""
    trains = {int(u): np.sort(sorting.get_unit_spike_train(u).astype(np.int64)) for u in sorting.unit_ids}
    mass = Counter()
    rows = []
    for (frm, to), cnt in pair_counts.items():
        if cnt < min_pair or frm not in trains or to not in trains:
            mass["too_few"] += cnt
            continue
        v = ccg_verdict_pair(trains[frm], trains[to], win_frames=win_frames)
        mass[v["verdict"]] += cnt
        rows.append((frm, to, cnt, v["verdict"], v["ratio"], v["n_co"]))
    return mass, sorted(rows, key=lambda r: -r[2])


def run_mode(rec, sorting, windows, *, margin, use_tight, min_pair):
    win_frames = int(WIN_S * FS)
    nfr = rec.get_num_frames()
    pair_counts = Counter()
    n_total = n_reassigned = 0
    for h in windows:
        a = int(h * 3600 * FS)
        b = min(a + win_frames, nfr)
        if a >= nfr:
            continue
        spikes, cosines, bank_ids = window_assignment_cosines(rec, sorting, a, b, n_jobs=N_JOBS)
        if spikes is None:
            continue
        new_spikes, reassigned = competitive_reassign(None, None, spikes, margin=margin,
                                                      use_tight=use_tight, cosines=cosines)
        n_total += spikes.size
        n_reassigned += int(reassigned.sum())
        frm = bank_ids[spikes["cluster_index"][reassigned]]
        to = bank_ids[new_spikes["cluster_index"][reassigned]]
        for f, t in zip(frm, to):
            pair_counts[(int(f), int(t))] += 1
    mass, rows = classify_pairs(pair_counts, sorting, win_frames, min_pair=min_pair)
    return dict(n_total=n_total, n_reassigned=n_reassigned, pair_counts=pair_counts, mass=mass, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="assembled_reestimate")
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min-pair", type=int, default=20)
    args = ap.parse_args()

    rec = materialize_span(OUT, START_S, DUR_S)
    sorting = si.load(OUT / args.variant)
    print(f"variant {args.variant}: {sorting.get_num_units()} units; windows {args.windows_h}h; "
          f"margin {args.margin}; min-pair {args.min_pair}", flush=True)

    out = {}
    for use_tight in (False, True):
        mode = "tight (rA)" if use_tight else "full (rF)"
        r = run_mode(rec, sorting, args.windows_h, margin=args.margin, use_tight=use_tight,
                     min_pair=args.min_pair)
        out["tight" if use_tight else "full"] = r
        frac = 100 * r["n_reassigned"] / r["n_total"] if r["n_total"] else float("nan")
        m, tot = r["mass"], max(r["n_reassigned"], 1)
        print(f"\n=== {mode}: {r['n_reassigned']:,}/{r['n_total']:,} spikes reassigned ({frac:.1f}%) ===",
              flush=True)
        print("  reassigned-spike mass by (from->to) CCG verdict "
              "(distinct=real cross-unit FIX, duplicate=oversplit CHURN):")
        for v in ("distinct", "duplicate", "ambiguous", "SEGREGATED", "too_few"):
            if m.get(v):
                print(f"    {v:>10}: {m[v]:>8,} ({100 * m[v] / tot:5.1f}% of reassigned)", flush=True)
        print("  top reassignment pairs (from->to  n  verdict  ratio  co_win):")
        for frm, to, cnt, verdict, ratio, n_co in r["rows"][:10]:
            rr = f"{ratio:.2f}" if np.isfinite(ratio) else "nan"
            print(f"    u{frm}->u{to}  n={cnt:<6} {verdict:<10} ratio={rr} co={n_co}", flush=True)

    summ = {mode: {"n_total": r["n_total"], "n_reassigned": r["n_reassigned"], "mass": dict(r["mass"])}
            for mode, r in out.items()}
    out_json = OUT / f"competitive_reassign_{args.variant}.json"
    out_json.write_text(json.dumps({"variant": args.variant, "modes": summ}, indent=2))
    print(f"\nwrote {out_json}\nDONE", flush=True)


if __name__ == "__main__":
    main()

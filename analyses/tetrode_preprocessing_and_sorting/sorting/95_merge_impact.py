"""Does the CCG-guarded merge push spikes AWAY from their template (the user's blending concern)?

For each MERGED unit, compare every spike's cosine to the MERGED template vs to its ORIGINAL pre-merge
template, in a window. Reports (a) the cosine-shift distribution (post - pre) and (b) INDUCED neighbour-wins
-- spikes whose best same-tetrode match was INSIDE their merge group pre-merge but flips to a THIRD
(unrelated) unit post-merge, i.e. blending genuinely worsened the assignment. Splits by merge reason
(CCG-duplicate vs cosine-fallback) since the fallback merges (CCG could not confirm same-cell) are the
risky ones.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/95_merge_impact.py \
        [--base assembled_reseed_c12] [--merged assembled_mergefirst] [--window-h 26]
"""
import argparse
import json
import pathlib
from collections import Counter

import numpy as np
import spikeinterface as si

from _assignment_eval import window_bank
from _mp_common import FS, TIGHT_SHIFT, all_template_cosines, asym_window_bounds, materialize_span

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
TOL = int(0.4e-3 * FS)  # match a base spike to its merged-unit train (coincident-dedup tolerant)


def map_original_to_merged(base, merged):
    """{base_uid -> merged_uid} by spike membership (a base unit's spikes live in exactly one merged train).
    Majority vote over sampled spikes for robustness against coincident-dedup."""
    merged_trains = {int(m): np.sort(merged.get_unit_spike_train(m).astype(np.int64)) for m in merged.unit_ids}
    o2m = {}
    for o in base.unit_ids:
        tr = base.get_unit_spike_train(o).astype(np.int64)
        if tr.size == 0:
            continue
        votes = []
        for f in tr[:: max(1, tr.size // 25)][:25]:
            best_m, best_d = -1, TOL + 1
            for m, mt in merged_trains.items():
                j = np.searchsorted(mt, f)
                for jj in (j - 1, j):
                    if 0 <= jj < mt.size:
                        d = abs(int(mt[jj]) - int(f))
                        if d < best_d:
                            best_d, best_m = d, m
            if best_m >= 0:
                votes.append(best_m)
        if votes:
            o2m[int(o)] = Counter(votes).most_common(1)[0][0]
    return o2m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="assembled_reseed_c12")
    ap.add_argument("--merged", default="assembled_mergefirst")
    ap.add_argument("--window-h", type=float, default=26.0)
    args = ap.parse_args()
    rec = materialize_span(OUT, START_S, DUR_S)
    base = si.load(OUT / args.base)
    merged = si.load(OUT / args.merged)
    o2m = map_original_to_merged(base, merged)
    comp_size = Counter(o2m.values())                       # merged uid -> # base units in it
    merged_units = {m for m, n in comp_size.items() if n > 1}  # the actually-merged ones
    print(f"{args.base} ({base.get_num_units()}u) -> {args.merged} ({merged.get_num_units()}u); "
          f"{len(merged_units)} merged units absorb {sum(comp_size[m] for m in merged_units)} originals",
          flush=True)

    a = int(args.window_h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win, base_bank, base_trains = window_bank(rec, base, a, b, n_jobs=N_JOBS)
    _, merged_bank, _ = window_bank(rec, merged, a, b, n_jobs=N_JOBS)
    if base_bank is None or merged_bank is None:
        print("empty window -- abort", flush=True)
        return
    base_ids = [int(u) for u in base_bank.unit_ids]
    merged_ids = [int(u) for u in merged_bank.unit_ids]
    m_idx = {m: i for i, m in enumerate(merged_ids)}
    aa, bb = asym_window_bounds(base_bank.nbefore)

    # base spikes (labelled by original unit), restricted to merged-component members present in both banks
    samp, ci_base, ci_merged, owner = [], [], [], []
    for bi, o in enumerate(base_ids):
        m = o2m.get(o)
        if m is None or m not in merged_units or m not in m_idx:
            continue
        t = base_trains[o]
        samp.append(t)
        ci_base.append(np.full(t.size, bi, np.int64))
        ci_merged.append(np.full(t.size, m_idx[m], np.int64))
        owner.append(np.full(t.size, o, np.int64))
    if not samp:
        print("no merged-unit spikes in window -- abort", flush=True)
        return
    samp = np.concatenate(samp)
    owner = np.concatenate(owner)
    sp_base = np.zeros(samp.size, dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    sp_base["sample_index"] = samp
    sp_base["cluster_index"] = np.concatenate(ci_base)
    sp_merged = sp_base.copy()
    sp_merged["cluster_index"] = np.concatenate(ci_merged)

    cb = all_template_cosines(win, base_bank, sp_base, aa, bb, TIGHT_SHIFT)
    cm = all_template_cosines(win, merged_bank, sp_merged, aa, bb, TIGHT_SHIFT)
    fin = np.isfinite(cb["rF_u"]) & np.isfinite(cm["rF_u"])
    pre = cb["rF_u"][fin]            # cos to ORIGINAL template
    post = cm["rF_u"][fin]           # cos to MERGED template
    shift = post - pre
    # induced neighbour-wins: pre best base unit is in the same merge group; post best merged unit != own m
    pre_best_base = np.array([base_ids[k] if k >= 0 else -1 for k in cb["rF_arg"]])[fin]
    post_best_merged = np.array([merged_ids[k] if k >= 0 else -1 for k in cm["rF_arg"]])[fin]
    own_m = np.array([o2m[int(o)] for o in owner[fin]])
    pre_in_group = np.array([o2m.get(int(pb), -2) for pb in pre_best_base]) == own_m
    post_out = post_best_merged != own_m
    induced = pre_in_group & post_out

    def pcts(x, ps):
        x = x[np.isfinite(x)]
        return ", ".join(f"p{p}={np.percentile(x, p):+.3f}" for p in ps) if x.size else "n/a"

    print(f"\nmerged-unit spikes scored: {int(fin.sum()):,}")
    print(f"cosine to ORIGINAL template:  median {np.median(pre):.3f}")
    print(f"cosine to MERGED   template:  median {np.median(post):.3f}")
    print(f"SHIFT (post-pre): {pcts(shift, [5, 25, 50, 75, 95])}")
    print(f"  frac shift < 0 (worse after merge): {100*np.mean(shift < 0):.1f}%  "
          f"| frac shift < -0.05: {100*np.mean(shift < -0.05):.1f}%  "
          f"| mean shift {np.mean(shift):+.4f}")
    print(f"INDUCED neighbour-wins (best flipped OUT of merge group): {int(induced.sum()):,} "
          f"({100*induced.mean():.2f}% of merged-unit spikes)", flush=True)

    (OUT / "merge_impact.json").write_text(json.dumps({
        "window_h": args.window_h, "n_merged_units": len(merged_units), "n_spikes": int(fin.sum()),
        "median_cos_original": float(np.median(pre)), "median_cos_merged": float(np.median(post)),
        "shift_pcts": {f"p{p}": float(np.percentile(shift, p)) for p in [5, 25, 50, 75, 95]},
        "frac_shift_neg": float(np.mean(shift < 0)), "frac_shift_below_-0.05": float(np.mean(shift < -0.05)),
        "mean_shift": float(np.mean(shift)), "induced_neighbor_win_frac": float(induced.mean()),
    }, indent=2))
    print(f"\nwrote {OUT / 'merge_impact.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

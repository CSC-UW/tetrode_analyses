"""Gallery: WHAT competitive reassignment moves, and WHY the CCG calls it fix / churn / ambiguous.

For the merge-first sorting, reassign spikes in a window (competitive_reassign), group the moves by
(from->to) unit pair, and for representative pairs in each CCG verdict bucket (distinct=FIX, duplicate=CHURN,
ambiguous) draw one row:
  * K example reassigned-spike SNIPPETS (black, 4 tetrode channels) with the FROM-unit template (gray dashed)
    and the TO-unit template (red dotted) overlaid -- so you can SEE whether the spike resembles `to` more.
    Each annotated with its peak MAD and cosine-to-from / cosine-to-to.
  * the CCG actually used for the verdict (from vs to, co-restricted to shared >=5-spike windows; +/-1.5 ms
    refractory zone shaded), annotated with the central/flank ratio and the verdict.
Row label = u{from}->u{to}, verdict. This lets the fix/churn/ambiguous call be judged by eye, per
"compare, don't assert; show pictures".

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/94_reassign_gallery.py \
        [--sorting assembled_mergefirst] [--window-h 26] [--use-tight] [--per-verdict 2] [--n-ex 3]
"""
import argparse
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si

from _assignment_eval import (MAXLAG_MS, REFR_MS, ccg_lags, ccg_verdict_pair, co_restrict, window_bank)
from _mp_common import (FS, TIGHT_SHIFT, _unit_groups_from_mask, all_template_cosines,
                        asym_window_bounds, competitive_reassign, materialize_span)

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
VERDICTS = [("distinct", "FIX (distinct cells)"), ("duplicate", "CHURN (same oversplit cell)"),
            ("ambiguous", "AMBIGUOUS")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sorting", default="assembled_mergefirst")
    ap.add_argument("--window-h", type=float, default=26.0)
    ap.add_argument("--use-tight", action="store_true", help="use the tight (rA) reassignment instead of full")
    ap.add_argument("--per-verdict", type=int, default=2)
    ap.add_argument("--n-ex", type=int, default=3)
    ap.add_argument("--min-pair", type=int, default=20)
    ap.add_argument("--select", choices=["cos", "mad"], default="cos",
                    help="pick example spikes by best cos->to (cos, default) or by amplitude (mad)")
    args = ap.parse_args()

    rec = materialize_span(OUT, START_S, DUR_S)
    sorting = si.load(OUT / args.sorting)
    a = int(args.window_h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win, bank, trains = window_bank(rec, sorting, a, b, n_jobs=N_JOBS)
    if bank is None:
        print("no unit >= template floor in window -- abort", flush=True)
        return
    bank_ids = np.asarray([int(u) for u in bank.unit_ids])
    samp, ci = [], []
    for i, u in enumerate(bank_ids):
        t = trains[int(u)]
        samp.append(t)
        ci.append(np.full(t.size, i, np.int64))
    spikes = np.zeros(int(sum(len(s) for s in samp)),
                      dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    spikes["sample_index"] = np.concatenate(samp)
    spikes["cluster_index"] = np.concatenate(ci)
    aa, bb = asym_window_bounds(bank.nbefore)
    cosines = all_template_cosines(win, bank, spikes, aa, bb, TIGHT_SHIFT)
    new_spikes, reassigned = competitive_reassign(None, None, spikes, margin=0.0,
                                                  use_tight=args.use_tight, cosines=cosines)

    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    rec_groups = np.asarray(win.get_property("group"))
    chan_ids = np.asarray(win.channel_ids)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    nbefore, n_samp, nfr = bank.nbefore, dense.shape[1], win.get_num_frames()
    rF_u = cosines["rA_u" if args.use_tight else "rF_u"]
    rF_best = cosines["rA_best" if args.use_tight else "rF_best"]
    mad = cosines["mad"]

    # group moves by (from_idx, to_idx)
    moves = defaultdict(list)
    ridx = np.flatnonzero(reassigned)
    for i in ridx:
        moves[(int(spikes["cluster_index"][i]), int(new_spikes["cluster_index"][i]))].append(int(i))
    # verdict per pair (on full trains) for pairs with enough mass
    full_trains = {int(u): np.sort(sorting.get_unit_spike_train(u).astype(np.int64)) for u in sorting.unit_ids}
    win_frames = int(WIN_S * FS)
    pair_rows = []
    for (fi, ti), idxs in moves.items():
        if len(idxs) < args.min_pair:
            continue
        fu, tu = int(bank_ids[fi]), int(bank_ids[ti])
        v = ccg_verdict_pair(full_trains[fu], full_trains[tu], win_frames=win_frames)
        pair_rows.append(dict(fi=fi, ti=ti, fu=fu, tu=tu, n=len(idxs), idxs=idxs, **v))
    # pick top per-verdict by moved count
    selected = []
    for vk, _ in VERDICTS:
        cands = sorted([p for p in pair_rows if p["verdict"] == vk], key=lambda p: -p["n"])
        selected += cands[:args.per_verdict]
    if not selected:
        print("no pairs met min-pair -- abort", flush=True)
        return
    print(f"{args.sorting} @ {args.window_h}h ({'tight' if args.use_tight else 'full'}): "
          f"{reassigned.sum():,} reassigned; showing {len(selected)} pairs", flush=True)

    K = args.n_ex
    fig, axes = plt.subplots(len(selected), K + 1, figsize=(2.15 * (K + 1), 2.1 * len(selected)),
                             squeeze=False)
    maxlag = int(MAXLAG_MS * 1e-3 * FS)
    edges = np.arange(-maxlag, maxlag + 1, int(0.5e-3 * FS)) / FS * 1000.0
    for r, p in enumerate(selected):
        chans = np.flatnonzero(rec_groups == ug[p["fi"]])
        tf = dense[p["fi"]][:, chans]
        tt = dense[p["ti"]][:, chans]
        key = (lambda i: -(rF_best[i] if np.isfinite(rF_best[i]) else -1)) if args.select == "cos" \
            else (lambda i: -(mad[i] if np.isfinite(mad[i]) else -1))
        ex = sorted(p["idxs"], key=key)[:K]
        for c in range(K):
            ax = axes[r][c]
            if c < len(ex):
                i = ex[c]
                off0 = int(spikes["sample_index"][i]) - nbefore
                if off0 < 0 or off0 + n_samp > nfr:
                    ax.axis("off")
                    continue
                tr = np.asarray(win.get_traces(start_frame=off0, end_frame=off0 + n_samp,
                                               channel_ids=list(chan_ids[chans])), dtype=np.float32)
                # overlay each template at its LEAST-SQUARES-FIT amplitude (cosine is scale-invariant,
                # so this is the honest shape comparison: residual = shape mismatch, not amplitude)
                af = float((tr * tf).sum() / max((tf * tf).sum(), 1e-9))
                at = float((tr * tt).sum() / max((tt * tt).sum(), 1e-9))
                # shared amplitude scale across all 4 channels -> preserves the RELATIVE cross-channel
                # pattern (the tetrode spatial signature). A faint baseline marks each channel so the
                # quiet channels (a tetrode spike is dominated by one wire) are visibly present, not absent.
                ymax = max(np.abs(tr).max(), 1.0)
                offv = 1.25 * ymax
                for ch in range(len(chans)):
                    ax.axhline(-ch * offv, color="0.88", lw=0.4, zorder=0)
                    ax.plot(tr[:, ch] - ch * offv, color="0.1", lw=0.9)
                    ax.plot(af * tf[:, ch] - ch * offv, color="0.55", lw=0.9, ls="--")
                    ax.plot(at * tt[:, ch] - ch * offv, color="#c0392b", lw=0.9, ls=":")
                ax.set_title(f"{mad[i]:.0f}MAD  cos→{p['fu']}:{rF_u[i]:.2f}\ncos→{p['tu']}:{rF_best[i]:.2f}",
                             fontsize=6.5)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"u{p['fu']}→u{p['tu']}\n{p['verdict']}", fontsize=8, rotation=0,
                              ha="right", va="center")
        # CCG panel
        axc = axes[r][K]
        ta, tb = full_trains[p["fu"]], full_trains[p["tu"]]
        nw = int(max(ta[-1] if ta.size else 0, tb[-1] if tb.size else 0) // win_frames) + 1
        co = np.flatnonzero((np.bincount(ta // win_frames, minlength=nw) >= 5)
                            & (np.bincount(tb // win_frames, minlength=nw) >= 5))
        lags = ccg_lags(co_restrict(tb, co, win_frames), co_restrict(ta, co, win_frames), maxlag)
        if lags.size:
            axc.hist(lags / FS * 1000.0, bins=edges, color="#3b6fb0", edgecolor="none")
        axc.axvspan(-REFR_MS, REFR_MS, color="red", alpha=0.18)
        rr = f"{p['ratio']:.2f}" if np.isfinite(p["ratio"]) else "nan"
        axc.text(0.98, 0.92, f"{p['verdict']}\nratio={rr}\nco_win={p['n_co']}\nn_moved={p['n']}",
                 transform=axc.transAxes, ha="right", va="top", fontsize=7,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        axc.set_yticks([])
        axc.tick_params(labelsize=6)
        if r == len(selected) - 1:
            axc.set_xlabel("CCG lag (ms) [to − from]", fontsize=7)
    fig.suptitle(f"Reassignment gallery — {args.sorting} @ {args.window_h}h "
                 f"({'tight rA' if args.use_tight else 'full rF'}): black=snippet, gray--=FROM-template, "
                 f"red:=TO-template;  right=CCG (red band=±1.5 ms refractory)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    tag = "tight" if args.use_tight else "full"
    p_out = OUT / f"reassign_gallery_{args.sorting}_{int(args.window_h)}h_{tag}.png"
    fig.savefig(p_out, dpi=140)
    plt.close(fig)
    print(f"wrote {p_out}\nDONE", flush=True)


if __name__ == "__main__":
    main()

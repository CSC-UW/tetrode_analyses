"""Admit-level gallery: what does each WOBBLE ADMIT level admit vs reject? (the amplitude-gate, visualized)

Companion to the cosine gallery (80, which fixed the admit and varied the shape gate r). Here we fix nothing
on the shape side and vary the WOBBLE ADMIT level f in {0.25..0.65} -- the amplitude/energy gate. wobble
detects where its objective `2*conv - ||t||^2 >= threshold` (wobble.py: objective_normalized = 2*objective -
norm_squared), and we set `threshold = f * tsq_median`. With a = conv/||t||^2 the per-spike amplitude scale,
that rule is EXACTLY `a >= a_f(u) = (1 + f*M/||t_u||^2)/2`  (M = tsq_median).

LABEL = DECISION (the fix vs the earlier version): we take ONE permissive POOL run (0.15x) for the candidate
spikes, compute each spike's a (per_spike_fit), and grade it admitted/rejected at level f by ITS OWN a vs
a_f(u). So every ADMITTED panel provably has a >= the bar and every REJECTED panel a < it -- the displayed
number determines the block. (The earlier version decided admit/reject from separate per-level wobble runs while
labelling from the pool run, so a junk pool fit could land in the admitted block whenever a real detection was
nearby. That decoupling is gone.) Caveat: a and r are computed at the integer detection alignment over the
unit's channels, so they are faithful estimates of -- not bit-identical to -- wobble's internal jittered fit;
and this clean re-grading shows the THRESHOLD RULE, not run-to-run greedy-peeling interactions.

Each row = one admit level; left block = ADMITTED examples, right block = REJECTED examples; panels = the 4-ch
snippet (solid) over the template (gray dashed), titled MAD(at the spike) / a / r. The story: as the level
rises, the REJECTED block fills with template-shaped low-amplitude spikes (high r, low a) -- the real spikes an
amplitude gate drops (the dropout), which the scale-invariant cosine gate keeps.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/82_wobble_admit_gallery.py \
        [--window-h 26] [--units 2] [--levels 0.25 0.35 0.45 0.55 0.65] [--pool-factor 0.15]
"""
import argparse
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.core import get_noise_levels

from _mp_common import (_unit_groups_from_mask, build_templates_object, materialize_span,
                        per_spike_fit, run_matching, tsq_median, wobble_method_kwargs)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
DETECT_THRESH = 5.5
NBEFORE, NAFTER = 30, 60
PEAK_HALF = 15   # MAD measured within +/-0.5 ms of the detection point (the spike), not the whole window
ADM_COLOR, REJ_COLOR = "#2e8b57", "#d8743b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-h", type=float, default=26.0)
    ap.add_argument("--units", type=int, default=2, help="units to plot (top by pool spike count)")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55, 0.65])
    ap.add_argument("--pool-factor", type=float, default=0.15, help="permissive admit for the candidate pool")
    ap.add_argument("--n-ex", type=int, default=5, help="example panels per admitted/rejected block")
    args = ap.parse_args()
    levels = sorted(args.levels)
    rng = np.random.default_rng(0)
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)
    a0 = int(args.window_h * 3600 * FS)
    b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a0, b0)
    win.reset_times()
    noise = get_noise_levels(win, return_in_uV=False)
    sdir = WV / "admit_gallery" / f"w{int(args.window_h)}h_ref"
    shutil.rmtree(sdir, ignore_errors=True)
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore_t = bank.nbefore
    med = tsq_median(bank)
    ids = np.asarray(bank.unit_ids)

    # single permissive pool run + per-spike (a, r); admit/reject is re-graded per level from a (label=decision)
    _, pool = run_matching(win, bank, method="wobble", n_jobs=N_JOBS,
                           method_kwargs=wobble_method_kwargs(bank, threshold=args.pool_factor * med))
    ps_all = pool["sample_index"].astype(np.int64)
    pci_all = pool["cluster_index"].astype(np.int64)
    a_all, r_all = per_spike_fit(pool, bank, win)
    print(f"window @ {args.window_h:.0f}h: pool {args.pool_factor}x -> {ps_all.size:,} candidate spikes; "
          f"levels {levels}; admit rule a >= (1 + f*M/||t||^2)/2", flush=True)

    counts = np.bincount(pci_all, minlength=ids.size)
    picked = [u for u in np.argsort(counts)[::-1][:args.units] if counts[u] > 0]

    for u_idx in picked:
        g = int(ug[u_idx])
        chans = np.flatnonzero(rec_groups == g)
        templ = dense[u_idx][:, chans]
        tsq_u = float((templ ** 2).sum())
        sel = pci_all == u_idx
        ps_u, a_u, r_u = ps_all[sel], a_all[sel], r_all[sel]
        finite = np.isfinite(a_u) & np.isfinite(r_u)

        def draw(ax, k, color):
            sample = int(ps_u[k])
            tr = np.asarray(win.get_traces(start_frame=sample - NBEFORE, end_frame=sample + NAFTER,
                                           channel_ids=list(chan_ids[chans])), dtype=np.float32)
            peak = tr[NBEFORE - PEAK_HALF:NBEFORE + PEAK_HALF]          # tight window around the spike
            mad = float(np.max(np.abs(peak) / noise[chans][None, :]))   # amplitude AT the spike, not window-max
            ymax = max(np.abs(tr).max(), np.abs(templ).max(), 1.0)
            off = 1.3 * ymax
            xt = np.arange(templ.shape[0]) + (NBEFORE - nbefore_t)
            for c in range(len(chans)):
                ax.plot(tr[:, c] - c * off, color=color, lw=0.9)
                ax.plot(xt, templ[:, c] - c * off, color="0.45", lw=0.8, ls="--")
            ax.axvline(NBEFORE, color="0.8", lw=0.6, zorder=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-3.4 * off, 1.4 * off)
            ax.set_title(f"{mad:.1f}MAD a{a_u[k]:.2f} r{r_u[k]:.2f}", fontsize=6.5,
                         color=("#c0392b" if mad < DETECT_THRESH else "0.1"))

        ne = args.n_ex
        fig, axes = plt.subplots(len(levels), 2 * ne, figsize=(2.0 * 2 * ne, 1.9 * len(levels)), squeeze=False)
        for ri, f in enumerate(levels):
            a_f = (1 + f * med / tsq_u) / 2 if tsq_u > 0 else float("nan")
            adm_idx = np.flatnonzero(finite & (a_u >= a_f))   # wobble's exact rule on THIS spike's a
            rej_idx = np.flatnonzero(finite & (a_u < a_f))
            for blk, (idxs, color) in enumerate(((adm_idx, ADM_COLOR), (rej_idx, REJ_COLOR))):
                pick = rng.choice(idxs, size=min(ne, idxs.size), replace=False) if idxs.size else np.empty(0, int)
                for c in range(ne):
                    ax = axes[ri][blk * ne + c]
                    if c < pick.size:
                        draw(ax, int(pick[c]), color)
                    else:
                        ax.axis("off")
            axes[ri][0].set_ylabel(f"{f:.2f}x\na>={a_f:.2f}\nadm {adm_idx.size}\nrej {rej_idx.size}",
                                   fontsize=8, rotation=0, ha="right", va="center")
        fig.text(0.30, 0.965, "ADMITTED (a >= bar)", ha="center", fontsize=11, color=ADM_COLOR)
        fig.text(0.74, 0.965, "REJECTED (a < bar)", ha="center", fontsize=11, color=REJ_COLOR)
        fig.suptitle(f"Wobble admit-level gallery u{ids[u_idx]} (tet {g}) @ {args.window_h:.0f}h -- rows = admit "
                     f"level f; admit iff a >= a_f = (1 + f*M/||t||^2)/2; snippet (solid) vs template (dashed)\n"
                     f"as f rises the bar rises -> REJECTED fills with template-shaped low-a spikes (the "
                     f"amplitude-gate dropout the cosine gate keeps)", fontsize=9.5, y=0.99)
        fig.tight_layout(rect=(0.05, 0, 1, 0.95))
        p = WV / f"admit_gallery_u{ids[u_idx]}_w{int(args.window_h)}h.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  wrote {p.name} (pool n={int(counts[u_idx])}, tsq_u/M={tsq_u / med:.2f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

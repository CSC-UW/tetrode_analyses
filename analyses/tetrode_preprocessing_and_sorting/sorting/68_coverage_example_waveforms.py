"""Example detected-event waveforms per amplitude (MAD) band, unclaimed vs claimed (tetrode coverage).

Grounds the spike-coverage tables (scripts 64/66): for each MAD band, pull several real detected events
from the materialized binary and plot their 4-channel (tetrode) snippets, split into UNCLAIMED (orange,
no unit within +/-0.5 ms) and CLAIMED (blue). Lets you eyeball whether low-MAD unclaimed events are
noise/MUA hash (expected) and whether high-MAD unclaimed events are real spikes the MP bank dropped.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/68_coverage_example_waveforms.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _mp_common import materialize_span

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
NBEFORE, NAFTER = 30, 60  # 1 ms before, 2 ms after
BANDS = [(5.5, 7), (7, 9), (9, 12), (12, 16), (16, 24), (24, np.inf)]
K = 4  # examples per (band, claimed/unclaimed)


def main():
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)
    nfr = rec.get_num_frames()
    d = np.load(OUT / "spike_coverage.npz")
    ps, pg, amp, claimed = (d["peak_sample"].astype(np.int64), d["peak_group"].astype(np.int64),
                            d["amp_mad"], d["claimed_0"])
    safe = (ps >= NBEFORE) & (ps < nfr - NAFTER)
    rng = np.random.default_rng(0)

    def snippet(i):
        g = int(pg[i])
        chans = np.flatnonzero(rec_groups == g)
        tr = rec.get_traces(start_frame=int(ps[i]) - NBEFORE, end_frame=int(ps[i]) + NAFTER,
                            channel_ids=list(chan_ids[chans]))
        return np.asarray(tr, dtype=np.float32)  # (T, 4)

    ncol = 2 * K
    fig, axes = plt.subplots(len(BANDS), ncol, figsize=(2.0 * ncol, 2.0 * len(BANDS)), squeeze=False)
    for r, (lo, hi) in enumerate(BANDS):
        inband = safe & (amp >= lo) & (amp < hi)
        cells = []  # (col, is_claimed, idx)
        for cj, want_claimed in enumerate([False] * K + [True] * K):
            pool = np.flatnonzero(inband & (claimed == want_claimed))
            cells.append((cj, want_claimed, int(rng.choice(pool)) if pool.size else None))
        snips = {cj: snippet(i) for cj, _, i in cells if i is not None}
        ymax = max((np.abs(s).max() for s in snips.values()), default=1.0)
        off = 1.3 * ymax
        for cj, want_claimed, i in cells:
            ax = axes[r][cj]
            ax.set_xticks([])
            ax.set_yticks([])
            if i is None:
                ax.text(0.5, 0.5, "none", ha="center", va="center", fontsize=8, color="0.6",
                        transform=ax.transAxes)
                continue
            s = snips[cj]
            color = "#3b7dd8" if want_claimed else "#d8743b"
            for ch in range(s.shape[1]):
                ax.plot(s[:, ch] - ch * off, color=color, lw=0.9)
            ax.axvline(NBEFORE, color="0.8", lw=0.6, zorder=0)
            ax.set_ylim(-3.4 * off, 1.3 * off)
            ax.set_title(f"{amp[i]:.0f} MAD", fontsize=8)
            if cj == 0:
                ax.set_ylabel(f"{lo:g}-{hi:g}\nMAD" if np.isfinite(hi) else f">{lo:g}\nMAD",
                              fontsize=9, rotation=0, ha="right", va="center")
            if r == 0:
                ax.text(0.5, 1.32, "UNCLAIMED" if not want_claimed and cj == 0 else
                        ("CLAIMED" if want_claimed and cj == K else ""), transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=10, fontweight="bold",
                        color="#d8743b" if not want_claimed else "#3b7dd8")
    fig.suptitle("Tetrode detected events by amplitude band: UNCLAIMED (orange) vs CLAIMED (blue), "
                 "4 channels stacked\nlow MAD = noise/MUA hash; high MAD = real spikes (unclaimed = MP "
                 "dropout / overlap)", fontsize=11)
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    p = OUT / "coverage_example_waveforms.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()

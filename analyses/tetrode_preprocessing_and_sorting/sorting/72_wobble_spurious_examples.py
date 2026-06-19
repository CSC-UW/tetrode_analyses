"""Show what "spurious" matcher spikes actually look like (wobble vs circus-omp), so we can judge whether
they are real-but-undetected events or noise fits.

A "spurious" spike (over-detection metric, script 71 / _wobble_eval.spurious_fraction) = a matcher output
spike with NO locally-exclusive negative >=5.5 MAD peak within +/-0.5 ms on the same tetrode. That reference
detector is sorter-agnostic (NOT MS5's or the matchers' own front-end), so "spurious" conflates three cases:
  (a) sub-threshold recovery  -- a real spike below 5.5 MAD the matcher fit by scaling its template down
  (b) collision suppression   -- a real spike the locally_exclusive rule dropped (a bigger peak within radius)
  (c) noise / residual fit    -- a genuine false positive
This script can't fully separate them, but plotting the 4-ch snippet with the ASSIGNED unit's template
overlaid + the snippet's peak amplitude (MAD) lets you eyeball it: a clear template-shaped deflection (even
if small => sub-threshold, case a) vs incoherent noise (case c). It also reports the sub-threshold (<5.5 MAD)
vs supra-threshold (>=5.5 MAD = case b / near-miss) split of a random sample of each matcher's spurious spikes.

Uses h=26 (the largest spurious gap: circus ~19%, wobble ~27%). Pre-dedup matcher output so cluster_index
maps directly to a bank template for the overlay (dedup doesn't change which spikes are spurious).

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes \
        python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/72_wobble_spurious_examples.py
"""
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.core import get_noise_levels

from _mp_common import (_unit_groups_from_mask, build_templates_object, materialize_span,
                        run_matching, wobble_method_kwargs)
from _wobble_eval import _within_tol, detect_window_peaks, tsq_median
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_H, WIN_S = 26.0, 1800.0
N_JOBS = 16
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}
DETECT_THRESH = 5.5
TOL = int(0.5e-3 * FS)
NBEFORE, NAFTER = 30, 60      # 1 ms before, 2 ms after the spike sample
SUBSAMPLE = 3000             # spurious spikes to characterize (MAD split) per matcher
N_EX = 8                     # spurious examples plotted per matcher
N_SUP = 8                    # supported (non-spurious) baseline examples per matcher


def main():
    factor = json.loads((WV / "threshold_calib_w26h.json").read_text())["chosen_factor"]
    rng = np.random.default_rng(0)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)
    a = int(WIN_H * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a, b)
    win.reset_times()
    nfr = win.get_num_frames()
    noise = get_noise_levels(win, return_in_uV=False)

    sdir = WV / "spurious" / "ref_sort"
    shutil.rmtree(sdir, ignore_errors=True)
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)  # per bank-unit group (cluster_index order)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)  # (n_units, n_samp, 64)
    nbefore_t = bank.nbefore
    thr_h = factor * tsq_median(bank)
    peak_s, peak_g, _ = detect_window_peaks(win, n_jobs=N_JOBS)
    pbt = {int(g): np.sort(peak_s[peak_g == g]) for g in np.unique(peak_g)}
    print(f"window @ {WIN_H:.0f}h: {bank.unit_ids.size} units; wobble thr={thr_h:.4g}; "
          f"{peak_s.size:,} detected events\n", flush=True)

    def snippet(sample, g):
        chans = np.flatnonzero(rec_groups == g)
        tr = win.get_traces(start_frame=sample - NBEFORE, end_frame=sample + NAFTER,
                            channel_ids=list(chan_ids[chans]))
        return np.asarray(tr, dtype=np.float32), chans  # (T,4)

    def peak_mad(tr, chans):
        return float(np.max(np.abs(tr) / noise[chans][None, :]))

    arms = {}
    for name, mk in (("circus-omp", CIRCUS_KW),
                     ("wobble", wobble_method_kwargs(bank, threshold=thr_h))):
        method = "circus-omp" if name == "circus-omp" else "wobble"
        _, spikes = run_matching(win, bank, method=method, method_kwargs=mk, n_jobs=N_JOBS)
        s = spikes["sample_index"].astype(np.int64)
        ci = spikes["cluster_index"].astype(np.int64)
        sg = ug[ci]
        inb = (s >= NBEFORE) & (s < nfr - NAFTER)
        supported = _within_tol(s, sg, pbt, TOL)
        spur_idx = np.flatnonzero(~supported & inb)
        sup_idx = np.flatnonzero(supported & inb)
        spur_frac = float((~supported).mean())

        # characterize a random subsample of spurious spikes by snippet peak MAD (sub- vs supra-threshold)
        sub = rng.choice(spur_idx, size=min(SUBSAMPLE, spur_idx.size), replace=False)
        mad_sub = np.array([peak_mad(*snippet(int(s[i]), int(sg[i]))) for i in sub])
        frac_subthr = float(np.mean(mad_sub < DETECT_THRESH))
        print(f"{name}: spurious={spur_frac*100:.1f}% of {s.size:,} spikes | of a {sub.size}-sample of "
              f"spurious: {frac_subthr*100:.0f}% are <5.5 MAD (sub-threshold, detector-invisible), "
              f"{(1-frac_subthr)*100:.0f}% are >=5.5 MAD (supra: collision-suppressed / near-miss); "
              f"sample MAD median={np.median(mad_sub):.1f}", flush=True)

        ex_spur = rng.choice(spur_idx, size=min(N_EX, spur_idx.size), replace=False)
        ex_sup = rng.choice(sup_idx, size=min(N_SUP, sup_idx.size), replace=False)
        arms[name] = dict(s=s, ci=ci, sg=sg, ex_spur=ex_spur, ex_sup=ex_sup, spur_frac=spur_frac,
                          frac_subthr=frac_subthr)

    # ---- plot: rows = [circus spurious, wobble spurious]; baseline supported in a 2nd figure ----
    def draw(ax, idx, arm, color):
        s, ci, sg = arm["s"], arm["ci"], arm["sg"]
        sample, g, unit_i = int(s[idx]), int(sg[idx]), int(ci[idx])
        tr, chans = snippet(sample, g)
        mad = peak_mad(tr, chans)
        templ = dense[unit_i][:, chans]  # (n_samp_t, 4)
        ymax = max(np.abs(tr).max(), np.abs(templ).max(), 1.0)
        off = 1.3 * ymax
        xt = np.arange(templ.shape[0]) + (NBEFORE - nbefore_t)
        for c in range(4):
            ax.plot(tr[:, c] - c * off, color=color, lw=0.9)
            ax.plot(xt, templ[:, c] - c * off, color="0.45", lw=0.8, ls="--")
        ax.axvline(NBEFORE, color="0.8", lw=0.6, zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-3.4 * off, 1.4 * off)
        sub = mad < DETECT_THRESH
        ax.set_title(f"u{bank.unit_ids[unit_i]} tet{g} {mad:.1f}MAD" + (" sub" if sub else " supra"),
                     fontsize=7.5, color=("#c0392b" if sub else "0.1"))

    for tag, key, npan in (("spurious", "ex_spur", N_EX), ("supported", "ex_sup", N_SUP)):
        fig, axes = plt.subplots(2, npan, figsize=(2.0 * npan, 4.6), squeeze=False)
        for r, (name, color) in enumerate((("circus-omp", "#d8743b"), ("wobble", "#3b7dd8"))):
            arm = arms[name]
            for c in range(npan):
                ax = axes[r][c]
                if c < len(arm[key]):
                    draw(ax, int(arm[key][c]), arm, color)
                else:
                    ax.axis("off")
            axes[r][0].set_ylabel(f"{name}\n({arm['spur_frac']*100:.0f}% spur)", fontsize=9,
                                  color=color, rotation=0, ha="right", va="center")
        ttl = ("SPURIOUS spikes (no >=5.5 MAD peak within +/-0.5 ms): snippet (solid) vs assigned template "
               "(gray dashed); red title = sub-threshold" if tag == "spurious" else
               "SUPPORTED spikes (baseline: a detected peak IS within +/-0.5 ms)")
        fig.suptitle(f"{ttl}\nh={WIN_H:.0f}h, 4 channels stacked", fontsize=10)
        fig.tight_layout(rect=(0.04, 0, 1, 0.93))
        p = WV / f"{tag}_examples.png"
        fig.savefig(p, dpi=135)
        plt.close(fig)
        print(f"wrote {p}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

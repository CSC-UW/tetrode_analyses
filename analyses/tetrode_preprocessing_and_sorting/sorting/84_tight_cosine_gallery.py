"""Tight-window cosine gallery: SHOW the spikes the tight+shift cosine drops that the full-window cosine keeps.

The frontier test (script 83) found that at MATCHED precision (median rp <= 0.01) a tight cosine window claims
FEWER detected 5.5-10 MAD events than the full window. But "fewer detected events" is NOT "fewer real spikes":
near-threshold detections are a mix of genuine low-amplitude SUA + MUA/hash, and refractory contamination
(the precision metric) does not catch non-refractory MUA. So the verdict "tightening hurts" only holds if the
DROPPED events are real template-shaped spikes; if they are noise, tightening was correctly cleaning. cov_low
can't decide -- only the waveforms can. This gallery shows them.

Method: one permissive wobble pool (0.25x) on one window. Compute three cosines per spike vs its assigned
template -- rF = full window (-1.0..+2.0 ms, the production gate), rA0 = asym tight window (-0.3..+0.8 ms),
rA2 = asym window with shift-tolerance (max over +/-2 samples, the tight+shift variant). Find each variant's
MATCHED-precision r* on THIS window (lowest r* with median rp <= 0.01). Then for a unit, split its candidate
spikes into:
  * DROPPED-BY-TIGHTENING: rF >= r*_full AND rA2 < r*_asym  (full keeps, tight+shift drops -- the disputed set)
  * KEPT-BY-BOTH:          rF >= r*_full AND rA2 >= r*_asym  (reference: what the tight gate keeps)
and draw example snippets+template overlays in each MAD band (one figure per band), with the asym window shaded
so you can see WHAT the tight cosine is scoring. If the dropped panels look as template-shaped as the kept ones,
tightening dropped real spikes (verdict holds); if they look like noise, tightening was cleaning (verdict wrong).

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/84_tight_cosine_gallery.py \
        [--window-h 27] [--units 2] [--n-ex 6] [--pool-factor 0.25] [--rp-target 0.01]
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
                        run_matching, tsq_median, wobble_method_kwargs)
from _wobble_eval import detect_window_peaks, score_kept_spikes
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
ASYM_MS = (-0.3, 0.8)        # the tight window tested in script 83 (samples [21,54] at 30 kHz)
S_MAX = 2                     # shift-tolerance radius for the tight+shift variant
MAD_BANDS = [(5.5, 7.0), (7.0, 9.0), (9.0, 10.0), (10.0, 14.0)]
R_GRID = np.round(np.arange(0.40, 0.801, 0.025), 3)
PEAK_HALF = 15               # MAD measured within +/-0.5 ms of the detection point
DROP_C, KEEP_C = "#d8743b", "#2e8b57"


def three_variant_cosines(win, bank, spikes, asym, s_max):
    """Per-spike (rF, rA0, rA2, mad): full-window cosine, asym-window cosine, asym+/-shift cosine, MAD@spike."""
    rec_groups = np.asarray(win.get_property("group"))
    chan_ids = np.asarray(win.channel_ids)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore, n_samp = bank.nbefore, dense.shape[1]
    nfr = win.get_num_frames()
    noise = get_noise_levels(win, return_in_uV=False)
    s = spikes["sample_index"].astype(np.int64)
    ci = spikes["cluster_index"].astype(np.int64)
    g_all = ug[ci]
    off = s - nbefore
    valid = (off - s_max >= 0) & (off + n_samp + s_max <= nfr)
    ext_cols = np.arange(-s_max, n_samp + s_max)
    a, b = asym
    rF = np.full(s.size, np.nan)
    rA0 = np.full(s.size, np.nan)
    rA2 = np.full(s.size, np.nan)
    mad = np.full(s.size, np.nan)
    for g in np.unique(g_all):
        on_g = np.flatnonzero((g_all == g) & valid)
        if on_g.size == 0:
            continue
        chans = np.flatnonzero(rec_groups == g)
        tr = np.asarray(win.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)
        nz = noise[chans]
        for u in np.unique(ci[on_g]):
            sel = on_g[ci[on_g] == u]
            ext = tr[off[sel][:, None] + ext_cols[None, :], :]          # (n, n_samp+2*s_max, 4)
            templ = dense[u][:, chans]
            sF = ext[:, s_max:s_max + n_samp, :]
            tsqF = float((templ ** 2).sum())
            dF = np.sqrt(tsqF * np.einsum("ntc,ntc->n", sF, sF))
            rF[sel] = np.where(dF > 0, np.einsum("ntc,tc->n", sF, templ) / dF, np.nan)
            tA = templ[a:b, :]
            tsqA = float((tA ** 2).sum())
            best = None
            for k in range(-s_max, s_max + 1):
                sk = ext[:, s_max + a + k:s_max + b + k, :]
                dk = np.sqrt(tsqA * np.einsum("ntc,ntc->n", sk, sk))
                rk = np.where(dk > 0, np.einsum("ntc,tc->n", sk, tA) / dk, np.nan)
                if k == 0:
                    rA0[sel] = rk
                best = rk if best is None else np.fmax(best, rk)
            rA2[sel] = best
            peak = sF[:, nbefore - PEAK_HALF:nbefore + PEAK_HALF, :]
            mad[sel] = np.max(np.abs(peak) / nz[None, None, :], axis=(1, 2))
    return rF, rA0, rA2, mad


def matched_rstar(r, s, ci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, target):
    """Lowest r* in R_GRID with median rp <= target (matched precision); None if never clean."""
    for rr in R_GRID:
        keep = r >= rr
        sc = score_kept_spikes(s[keep], ci[keep], ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)
        if np.isfinite(sc["median_rp"]) and sc["median_rp"] <= target:
            return float(rr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-h", type=float, default=27.0)
    ap.add_argument("--units", type=int, default=2, help="units to gallery (top by pool spike count)")
    ap.add_argument("--n-ex", type=int, default=6, help="example panels per block per band")
    ap.add_argument("--pool-factor", type=float, default=0.25)
    ap.add_argument("--rp-target", type=float, default=0.01)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)
    a0 = int(args.window_h * 3600 * FS)
    b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a0, b0)
    win.reset_times()
    nfr = win.get_num_frames()
    noise = get_noise_levels(win, return_in_uV=False)
    sdir = WV / "tight_cosine_gallery" / f"w{int(args.window_h)}h_ref"
    shutil.rmtree(sdir, ignore_errors=True)
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore = bank.nbefore
    ids = np.asarray(bank.unit_ids)
    a = nbefore + int(round(ASYM_MS[0] * FS / 1000.0))
    b = nbefore + int(round(ASYM_MS[1] * FS / 1000.0))
    peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
    peak_by_tet = {int(gg): np.sort(peak_s[peak_g == gg]) for gg in np.unique(peak_g)}

    _, pool = run_matching(win, bank, method="wobble", n_jobs=N_JOBS,
                           method_kwargs=wobble_method_kwargs(bank, threshold=args.pool_factor * tsq_median(bank)))
    ps = pool["sample_index"].astype(np.int64)
    pci = pool["cluster_index"].astype(np.int64)
    rF, rA0, rA2, mad = three_variant_cosines(win, bank, pool, (a, b), S_MAX)
    rstar_full = matched_rstar(rF, ps, pci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, args.rp_target)
    rstar_asym = matched_rstar(rA2, ps, pci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, args.rp_target)
    print(f"window @ {args.window_h:.0f}h: pool {args.pool_factor}x -> {pool.size:,} candidates; asym window "
          f"samples [{a},{b}] (-0.3..+0.8 ms), shift +/-{S_MAX}", flush=True)
    print(f"matched-precision r* (median rp <= {args.rp_target}): full|0 = {rstar_full} ; asym|+-{S_MAX} = "
          f"{rstar_asym}", flush=True)
    if rstar_full is None or rstar_asym is None:
        print("  a variant never reached the precision target on this window; aborting", flush=True)
        return

    finite = np.isfinite(rF) & np.isfinite(rA2) & np.isfinite(mad)
    dropped = finite & (rF >= rstar_full) & (rA2 < rstar_asym)   # full keeps, tight+shift drops -- disputed set
    kept = finite & (rF >= rstar_full) & (rA2 >= rstar_asym)     # both keep -- reference
    counts = np.bincount(pci[finite], minlength=ids.size)
    picked = [u for u in np.argsort(counts)[::-1][:args.units] if counts[u] > 0]

    def draw(ax, sample, chans, templ, color):
        tr = np.asarray(win.get_traces(start_frame=sample - nbefore, end_frame=sample - nbefore + templ.shape[0],
                                       channel_ids=list(chan_ids[chans])), dtype=np.float32)
        pk = tr[nbefore - PEAK_HALF:nbefore + PEAK_HALF]
        m = float(np.max(np.abs(pk) / noise[chans][None, :]))
        ymax = max(np.abs(tr).max(), np.abs(templ).max(), 1.0)
        off = 1.3 * ymax
        ax.axvspan(a, b, color="#cfe8ff", alpha=0.6, zorder=0)            # the asym (tight) window
        for c in range(len(chans)):
            ax.plot(tr[:, c] - c * off, color=color, lw=0.9)
            ax.plot(templ[:, c] - c * off, color="0.45", lw=0.8, ls="--")
        ax.axvline(nbefore, color="0.7", lw=0.6, zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-3.4 * off, 1.4 * off)
        return m

    for u_idx in picked:
        g = int(ug[u_idx])
        chans = np.flatnonzero(rec_groups == g)
        templ = dense[u_idx][:, chans]
        usel = pci == u_idx
        print(f"\nunit {ids[u_idx]} (tet {g}, pool n={int(counts[u_idx])}): dropped-by-tightening vs kept-by-both "
              f"per MAD band", flush=True)
        for lo, hi in MAD_BANDS:
            band = usel & (mad >= lo) & (mad < hi)
            d_idx = np.flatnonzero(band & dropped)
            k_idx = np.flatnonzero(band & kept)
            nb = int((band & finite).sum())
            fr = 100.0 * d_idx.size / nb if nb else float("nan")
            print(f"  MAD {lo:g}-{hi:g}: dropped {d_idx.size}, kept {k_idx.size}  "
                  f"(dropped = {fr:.0f}% of {nb} full-kept band events)", flush=True)
            ne = args.n_ex
            fig, axes = plt.subplots(2, ne, figsize=(2.0 * ne, 4.4), squeeze=False)
            for blk, (idxs, color, name) in enumerate(((d_idx, DROP_C, "DROPPED by tight+shift"),
                                                       (k_idx, KEEP_C, "KEPT by both"))):
                pick = rng.choice(idxs, size=min(ne, idxs.size), replace=False) if idxs.size else np.empty(0, int)
                for c in range(ne):
                    ax = axes[blk][c]
                    if c < pick.size:
                        j = int(pick[c])
                        m = draw(ax, int(ps[j]), chans, templ, color)
                        ax.set_title(f"{m:.1f}MAD\nrF{rF[j]:.2f} rA{rA0[j]:.2f} rA±{rA2[j]:.2f}",
                                     fontsize=6.5, color=color)
                    else:
                        ax.axis("off")
                axes[blk][0].set_ylabel(name, fontsize=8, color=color)
            fig.suptitle(f"u{ids[u_idx]} tet {g} @ {args.window_h:.0f}h  |  MAD {lo:g}-{hi:g}  |  shaded = asym "
                         f"tight window (-0.3..+0.8 ms)\nrF=full r* {rstar_full:.3f}  rA=asym  rA±=asym+shift "
                         f"r* {rstar_asym:.3f}   (top: full keeps but tight+shift drops; bottom: both keep)",
                         fontsize=8.5)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            p = WV / f"tight_cosine_gallery_u{ids[u_idx]}_w{int(args.window_h)}h_mad{lo:g}-{hi:g}.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"    wrote {p.name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

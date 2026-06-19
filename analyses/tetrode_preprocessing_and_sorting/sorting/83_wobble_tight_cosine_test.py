"""Phase A addendum: tight-window, shift-tolerant cosine -- worth implementing into the shape gate?

The production shape gate (per_spike_cosine) scores r = cos(snippet, template) over the FULL 90-sample
template (-1.0..+2.0 ms around the trough) at the INTEGER detection alignment. Two refinements were
proposed in the gallery discussion:
  * TIGHT WINDOW: restrict r to the discriminative peak region, dropping the ~0.7 ms of near-flat
    pre-peak baseline and the ~1.2 ms low-SNR AHP tail -- both dilute the cosine with non-unit-specific
    samples (the "does wobble care about the far tail" question).
  * SHIFT TOLERANCE: r = max over small integer shifts k, to undo integer-rounding misalignment
    (wobble fits sub-sample jitter internally; the integer-aligned cosine can depress a real spike that
    landed +/-1 sample off, for no true shape reason).

THE CAVEAT THAT GOVERNS THE READOUT: both refinements MONOTONICALLY raise r (fewer diluting samples;
max over more candidates). Holding r* fixed would therefore admit more spikes BY CONSTRUCTION -- that is
gate-loosening in disguise, which the no-loosening rule forbids. So every variant is evaluated at its
OWN re-pinned r* (the lowest r* that holds median rp_contamination <= --rp-target, i.e. MATCHED
PRECISION), and the win condition is MORE real low-amplitude coverage (5.5-10 MAD band) at that matched
precision -- not a higher r. A variant that does not beat the full-window fixed cosine on the
(rp, cov_low) frontier, or does not move r*, is not worth implementing.

Method: one permissive wobble POOL run per window (the shape gate, not the admit, is binding), then the
whole {window x shift} grid of r is a FREE post-hoc recompute on that pool (same structure as the
admit x r grid, script 81). circus-omp [0.8,inf] is plotted as the absolute B0 anchor.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/83_wobble_tight_cosine_test.py \
        [--windows-h 11 19 27 40] [--pool-factor 0.25] [--rp-target 0.01]
"""
import argparse
import json
import pathlib
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}
# cosine windows, as (name, t0_ms, t1_ms) relative to the trough; full = the current production r
WINDOWS_MS = [("full", -1.0, 2.0), ("wide", -0.5, 1.0), ("asym", -0.3, 0.8),
              ("sym75", -0.75, 0.75), ("tight", -0.27, 0.4)]
SHIFT_RADII = [0, 1, 2, 3]   # max-over-shift in [-R, R] samples; 0 = current (integer-aligned) cosine
S_MAX = max(SHIFT_RADII)
R_GRID = np.round(np.arange(0.40, 0.801, 0.025), 3)   # r* sweep to trace the precision-coverage frontier
BASE = ("full", 0)   # baseline variant = current production cosine (full window, no shift)


def win_bounds(t0_ms, t1_ms, nbefore, n_samp):
    a = nbefore + int(round(t0_ms * FS / 1000.0))
    b = nbefore + int(round(t1_ms * FS / 1000.0))
    return max(0, a), min(n_samp, b)


def windowed_shift_cosine(win, bank, spikes, windows_ab):
    """Per-spike cosine over a {window x shift} grid -> dict[(wname, radius)] -> r (NaN if snippet OOB).

    Mirrors _mp_common.per_spike_fit's per-tetrode/per-unit batching (loads only the group's channels),
    but extracts an extended snippet (+/- S_MAX samples) and, for each template sub-window [a,b], computes
    r at every integer shift k in [-S_MAX, S_MAX], then reduces to radius R = max over k in [-R, R].
    A degenerate (zero-norm) snippet window -> r = -1 (fails any gate); a fully OOB spike stays NaN.
    """
    rec_groups = np.asarray(win.get_property("group"))
    chan_ids = np.asarray(win.channel_ids)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore, n_samp = bank.nbefore, dense.shape[1]
    nfr = win.get_num_frames()
    s = spikes["sample_index"].astype(np.int64)
    ci = spikes["cluster_index"].astype(np.int64)
    g_all = ug[ci]
    off = s - nbefore
    valid = (off - S_MAX >= 0) & (off + n_samp + S_MAX <= nfr)
    ext_cols = np.arange(-S_MAX, n_samp + S_MAX)   # length n_samp + 2*S_MAX
    ks = np.arange(-S_MAX, S_MAX + 1)
    out = {(wn, R): np.full(s.size, np.nan) for (wn, _, _) in windows_ab for R in SHIFT_RADII}
    for g in np.unique(g_all):
        on_g = np.flatnonzero((g_all == g) & valid)
        if on_g.size == 0:
            continue
        chans = np.flatnonzero(rec_groups == g)
        tr = np.asarray(win.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)
        for u in np.unique(ci[on_g]):
            sel = on_g[ci[on_g] == u]
            ext = tr[off[sel][:, None] + ext_cols[None, :], :]          # (n, n_samp+2*S_MAX, 4)
            templ = dense[u][:, chans]                                  # (n_samp, 4)
            for (wn, a, b) in windows_ab:
                tw = templ[a:b, :]
                tsq = float((tw ** 2).sum())
                r_k = np.empty((sel.size, ks.size), dtype=np.float64)
                for j, k in enumerate(ks):
                    sw = ext[:, (a + k + S_MAX):(b + k + S_MAX), :]     # (n, W, 4)
                    conv = np.einsum("ntc,tc->n", sw, tw)
                    ssq = np.einsum("ntc,ntc->n", sw, sw)
                    denom = np.sqrt(tsq * ssq)
                    r_k[:, j] = np.where(denom > 0, conv / denom, -1.0)
                for R in SHIFT_RADII:
                    out[(wn, R)][sel] = r_k[:, S_MAX - R:S_MAX + R + 1].max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", type=float, nargs="+", default=[11.0, 19.0, 27.0, 40.0])
    ap.add_argument("--pool-factor", type=float, default=0.25, help="permissive admit for the candidate pool")
    ap.add_argument("--rp-target", type=float, default=0.01, help="matched-precision target (median rp)")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    variants = [(wn, R) for (wn, _, _) in WINDOWS_MS for R in SHIFT_RADII]
    results = {}
    for h in args.windows_h:
        a0 = int(h * 3600 * FS)
        b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a0, b0)
        win.reset_times()
        nfr = win.get_num_frames()
        sdir = WV / "tight_cosine" / f"w{int(h)}h_ref"
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.perf_counter()
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=args.n_jobs, seed=0)
        ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
        med = tsq_median(bank)
        nbefore, n_samp = bank.nbefore, int(bank.get_dense_templates().shape[1])
        windows_ab = [(wn, *win_bounds(t0_ms, t1_ms, nbefore, n_samp)) for (wn, t0_ms, t1_ms) in WINDOWS_MS]
        peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=args.n_jobs)
        peak_by_tet = {int(gg): np.sort(peak_s[peak_g == gg]) for gg in np.unique(peak_g)}
        print(f"\n=== @ {h:.0f}h: {bank.unit_ids.size} units, {(amp_mad >= 10).sum():,} >=10 MAD events, "
              f"{(amp_mad < 10).sum():,} low-amp (5.5-10), setup {time.perf_counter()-t0:.0f}s ===", flush=True)
        print("  windows (samples): " + " ".join(f"{wn}[{a},{b}]" for (wn, a, b) in windows_ab), flush=True)

        _, cs = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=args.n_jobs)
        circ = score_kept_spikes(cs["sample_index"].astype(np.int64), cs["cluster_index"].astype(np.int64),
                                 ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)
        print(f"  circus[0.8,inf] (B0 anchor): rp {circ['median_rp']:.4f} | cov10 {circ['cov10']:.1f}% | "
              f"cov_low {circ['cov_low']:.1f}% | n {circ['n_spikes']:,}", flush=True)

        tw = time.perf_counter()
        _, pool = run_matching(win, bank, method="wobble", n_jobs=args.n_jobs,
                               method_kwargs=wobble_method_kwargs(bank, threshold=args.pool_factor * med))
        sp_s = pool["sample_index"].astype(np.int64)
        sp_ci = pool["cluster_index"].astype(np.int64)
        grid = windowed_shift_cosine(win, bank, pool, windows_ab)
        print(f"  pool {args.pool_factor}x -> {pool.size:,} candidates; cosine grid "
              f"({len(variants)} variants) in {time.perf_counter()-tw:.0f}s", flush=True)

        wres = {"circus": circ}
        for (wn, R) in variants:
            r_all = grid[(wn, R)]
            curve = {}
            for rr in R_GRID:
                keep = r_all >= rr   # NaN -> False
                curve[f"{rr:.3f}"] = score_kept_spikes(sp_s[keep], sp_ci[keep], ug, peak_s, peak_g,
                                                       amp_mad, peak_by_tet, nfr)
            wres[f"{wn}|{R}"] = curve
        results[f"{h:.0f}h"] = wres
        # per-window matched-precision peek for the baseline + the primary tight window
        for vk in (f"{BASE[0]}|{BASE[1]}", "asym|2"):
            rp = np.array([wres[vk][f"{rr:.3f}"]["median_rp"] for rr in R_GRID])
            cl = np.array([wres[vk][f"{rr:.3f}"]["cov_low"] for rr in R_GRID])
            ok = np.flatnonzero(np.isfinite(rp) & (rp <= args.rp_target))
            if ok.size:
                i = int(ok[0])
                print(f"    {vk}: r*={R_GRID[i]:.3f} @rp<={args.rp_target} -> cov_low {cl[i]:.1f}%", flush=True)

    _summarize_and_plot(results, variants, args.rp_target)


def _agg(results, vk, field):
    """Median across windows of `field` over the r* grid for variant vk -> 1D [len(R_GRID)]."""
    hs = list(results)
    out = np.full(len(R_GRID), np.nan)
    for j, rr in enumerate(R_GRID):
        vals = [results[h][vk][f"{rr:.3f}"][field] for h in hs if vk in results[h]]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[j] = float(np.median(vals))
    return out


def _op_idx(rp_curve, target):
    """Index of the lowest r* (most permissive -> highest coverage) holding median rp <= target."""
    ok = np.flatnonzero(np.isfinite(rp_curve) & (rp_curve <= target))
    return int(ok[0]) if ok.size else None


def _summarize_and_plot(results, variants, rp_target):
    hs = list(results)
    circ_rp = float(np.median([results[h]["circus"]["median_rp"] for h in hs]))
    circ_cov10 = float(np.median([results[h]["circus"]["cov10"] for h in hs]))
    circ_covlow = float(np.median([results[h]["circus"]["cov_low"] for h in hs]))

    rows = {}   # vk -> dict with op-point metrics at matched precision
    for (wn, R) in variants:
        vk = f"{wn}|{R}"
        rp = _agg(results, vk, "median_rp")
        i = _op_idx(rp, rp_target)
        rows[vk] = {
            "rstar": float(R_GRID[i]) if i is not None else None,
            "cov10": float(_agg(results, vk, "cov10")[i]) if i is not None else float("nan"),
            "cov_low": float(_agg(results, vk, "cov_low")[i]) if i is not None else float("nan"),
            "spurious": float(_agg(results, vk, "spurious")[i]) if i is not None else float("nan"),
            "n_spikes": float(_agg(results, vk, "n_spikes")[i]) if i is not None else float("nan"),
        }
    base = rows[f"{BASE[0]}|{BASE[1]}"]
    print(f"\n=== matched-precision frontier @ median rp <= {rp_target} (median across {len(hs)} windows) ===",
          flush=True)
    print(f"  circus[0.8,inf] anchor: rp {circ_rp:.4f} | cov10 {circ_cov10:.1f}% | cov_low {circ_covlow:.1f}%",
          flush=True)
    print(f"  baseline full|0 (production r): r*={base['rstar']} cov10 {base['cov10']:.1f}% "
          f"cov_low {base['cov_low']:.1f}%", flush=True)
    print("  variant      r*     cov10   cov_low  d_cov_low  spurious", flush=True)
    order = sorted(variants, key=lambda v: -(rows[f"{v[0]}|{v[1]}"]["cov_low"]
                                             if np.isfinite(rows[f"{v[0]}|{v[1]}"]["cov_low"]) else -1))
    for (wn, R) in order:
        vk = f"{wn}|{R}"
        rw = rows[vk]
        rs = f"{rw['rstar']:.3f}" if rw["rstar"] is not None else "  -  "
        dlo = rw["cov_low"] - base["cov_low"]
        tag = "  <- baseline" if (wn, R) == BASE else ""
        print(f"  {vk:10s}  {rs}  {rw['cov10']:6.1f}  {rw['cov_low']:7.1f}  {dlo:+8.1f}  "
              f"{rw['spurious']:7.3f}{tag}", flush=True)
    best = order[0]
    bk = f"{best[0]}|{best[1]}"
    print(f"  -> best low-amp retention at matched precision: {bk} "
          f"(cov_low {rows[bk]['cov_low']:.1f}% vs baseline {base['cov_low']:.1f}%, "
          f"d={rows[bk]['cov_low'] - base['cov_low']:+.1f} pts; r* {rows[bk]['rstar']} vs {base['rstar']})",
          flush=True)

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    wnames = [wn for (wn, _, _) in WINDOWS_MS]
    wcolors = dict(zip(wnames, plt.cm.viridis(np.linspace(0, 0.85, len(wnames)))))

    def frontier(axis, vk, color, ls, label, field):
        rp = _agg(results, vk, "median_rp")
        cov = _agg(results, vk, field)
        o = np.argsort(rp)
        axis.plot(rp[o], cov[o], ls, color=color, lw=1.4, label=label)
        i = _op_idx(rp, rp_target)
        if i is not None:
            axis.plot(rp[i], cov[i], "o", color=color, ms=8, mfc="none", mew=1.6)

    # (A) window effect @ shift 0: cov10 vs rp
    for wn in wnames:
        frontier(ax[0, 0], f"{wn}|0", wcolors[wn], "-", wn, "cov10")
    ax[0, 0].plot(circ_rp, circ_cov10, "k*", ms=14, label="circus")
    ax[0, 0].axvline(rp_target, color="0.6", ls=":", lw=1)
    ax[0, 0].set(xlabel="median rp_contamination", ylabel=">=10 MAD coverage %",
                 title="(A) window effect @ shift 0 -- cov10 (o = r* @ matched precision)")
    ax[0, 0].legend(fontsize=8)
    # (B) window effect @ shift 0: cov_low vs rp (the metric that matters)
    for wn in wnames:
        frontier(ax[0, 1], f"{wn}|0", wcolors[wn], "-", wn, "cov_low")
    ax[0, 1].plot(circ_rp, circ_covlow, "k*", ms=14, label="circus")
    ax[0, 1].axvline(rp_target, color="0.6", ls=":", lw=1)
    ax[0, 1].set(xlabel="median rp_contamination", ylabel="low-amp (5.5-10 MAD) coverage %",
                 title="(B) window effect @ shift 0 -- cov_low (the low-amp retention frontier)")
    ax[0, 1].legend(fontsize=8)
    # (C) shift effect on cov_low: radius 0 (solid) vs radius 2 (dashed) for 3 windows
    for wn in ("full", "asym", "sym75"):
        frontier(ax[1, 0], f"{wn}|0", wcolors[wn], "-", f"{wn} shift0", "cov_low")
        frontier(ax[1, 0], f"{wn}|2", wcolors[wn], "--", f"{wn} shift+-2", "cov_low")
    ax[1, 0].axvline(rp_target, color="0.6", ls=":", lw=1)
    ax[1, 0].set(xlabel="median rp_contamination", ylabel="low-amp (5.5-10 MAD) coverage %",
                 title="(C) shift effect on cov_low (solid=shift0, dashed=+-2 samples)")
    ax[1, 0].legend(fontsize=8)
    # (D) summary bars: cov_low at matched precision, grouped by window, hue by shift radius
    x = np.arange(len(wnames))
    w = 0.2
    rcolors = dict(zip(SHIFT_RADII, plt.cm.plasma(np.linspace(0.1, 0.8, len(SHIFT_RADII)))))
    for ri, R in enumerate(SHIFT_RADII):
        vals = [rows[f"{wn}|{R}"]["cov_low"] for wn in wnames]
        ax[1, 1].bar(x + (ri - 1.5) * w, vals, w, color=rcolors[R], label=f"shift+-{R}")
    ax[1, 1].axhline(base["cov_low"], color="0.3", ls="--", lw=1.2, label="baseline (full|0)")
    ax[1, 1].axhline(circ_covlow, color="k", ls=":", lw=1.2, label="circus")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(wnames)
    ax[1, 1].set(xlabel="cosine window", ylabel="cov_low % @ matched precision",
                 title=f"(D) low-amp retention @ median rp <= {rp_target}")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("Tight-window, shift-tolerant cosine vs full-window fixed cosine -- at MATCHED precision "
                 "(re-pinned r*), does it recover more low-amp real spikes?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figp = WV / "tight_cosine_test.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)
    (WV / "tight_cosine_test.json").write_text(json.dumps(
        {"windows_ms": WINDOWS_MS, "shift_radii": SHIFT_RADII, "r_grid": R_GRID.tolist(),
         "rp_target": rp_target, "op_points": rows,
         "circus": {"rp": circ_rp, "cov10": circ_cov10, "cov_low": circ_covlow},
         "results": results}, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / 'tight_cosine_test.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

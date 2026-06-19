"""Do the tight-window-dropped spikes belong to the assigned unit, or to a same-tetrode NEIGHBOR?

The gallery (84) showed the dropped-by-tightening spikes are REAL (template-shaped, amplitude-independent drop).
But "real" != "belongs to THIS template": a neighbor's spike with similar gross shape can be assigned to u by
wobble (its objective `2*conv-||t||^2` matches on the FULL rank-4 template, NOT the trough), and the tight
trough-window cosine may be CORRECTLY rejecting it. If so, tightening improves unit SPECIFICITY (not real-spike
loss), and the pooled cov_low "loss" is an artifact of not re-offering the spike to the neighbor.

Decisive test: for each pool spike compute the cosine to EVERY same-tetrode template, under both the full window
and the tight+shift (asym -0.3..+0.8 ms, +/-2 samp) window. For the dropped-by-tightening set (rF_to_u >=
r*_full AND rA2_to_u < r*_asym), ask:
  * is the assigned unit u STILL the best tight-window match (just below threshold -> u's own variable spike),
    or does a NEIGHBOR win the tight window (-> mis-assignment the trough catches)?
  * if a neighbor wins, does ITS tight cosine CLEAR r*_asym (-> the spike WOULD be cleanly claimed by the
    neighbor = reassignment, NO real coverage loss) or not (-> matches nothing well = collision / genuine loss)?
Compared against the kept-by-both set as a control (should be dominated by u-is-best). Plus an example figure:
neighbor-wins dropped spikes with u's template (gray dashed) and the winning neighbor's template (red dotted).

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/85_tight_cosine_reassignment.py \
        [--window-h 27] [--units 2] [--rp-target 0.01]
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
ASYM_MS = (-0.3, 0.8)
S_MAX = 2
R_GRID = np.round(np.arange(0.40, 0.801, 0.025), 3)
PEAK_HALF = 15


def all_template_cosines(win, bank, spikes, a, b, s_max):
    """Per spike: (rF_u, rA_u to ASSIGNED unit; rF_best/arg, rA_best/arg over ALL same-tetrode templates; mad).

    rF = full-window cosine; rA = asym tight window with +/-s_max shift-tolerance (max over shifts). 'arg' are
    GLOBAL template indices. Mirrors per_spike_fit's per-tetrode batching; cosines to every co-tetrode template.
    """
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
    n = s.size
    rF_u = np.full(n, np.nan)
    rA_u = np.full(n, np.nan)
    rF_best = np.full(n, np.nan)
    rA_best = np.full(n, np.nan)
    rF_arg = np.full(n, -1, np.int64)
    rA_arg = np.full(n, -1, np.int64)
    mad = np.full(n, np.nan)
    for g in np.unique(g_all):
        on_g = np.flatnonzero((g_all == g) & valid)
        if on_g.size == 0:
            continue
        units_g = np.flatnonzero(ug == g)                  # global template indices on this tetrode
        chans = np.flatnonzero(rec_groups == g)
        nz = noise[chans]
        tr = np.asarray(win.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)
        ext = tr[off[on_g][:, None] + ext_cols[None, :], :]            # (m, n_samp+2*s_max, 4)
        Tf = dense[units_g][:, :, chans]                               # (nu, n_samp, 4)
        Ta = dense[units_g][:, a:b, chans]                             # (nu, W, 4)
        tsqF = np.einsum("jtc,jtc->j", Tf, Tf)
        tsqA = np.einsum("jwc,jwc->j", Ta, Ta)
        sF = ext[:, s_max:s_max + n_samp, :]
        ssqF = np.einsum("mtc,mtc->m", sF, sF)
        rF = np.einsum("mtc,jtc->mj", sF, Tf) / np.sqrt(ssqF[:, None] * tsqF[None, :])   # (m, nu)
        rA = np.full((on_g.size, units_g.size), -np.inf)
        for k in range(-s_max, s_max + 1):
            sk = ext[:, s_max + a + k:s_max + b + k, :]
            ssqk = np.einsum("mwc,mwc->m", sk, sk)
            rk = np.einsum("mwc,jwc->mj", sk, Ta) / np.sqrt(ssqk[:, None] * tsqA[None, :])
            rA = np.maximum(rA, rk)
        local_u = np.searchsorted(units_g, ci[on_g])                   # column of the assigned unit
        rows = np.arange(on_g.size)
        rF_u[on_g] = rF[rows, local_u]
        rA_u[on_g] = rA[rows, local_u]
        fb = np.argmax(rF, axis=1)
        ab = np.argmax(rA, axis=1)
        rF_best[on_g] = rF[rows, fb]
        rA_best[on_g] = rA[rows, ab]
        rF_arg[on_g] = units_g[fb]
        rA_arg[on_g] = units_g[ab]
        peak = sF[:, nbefore - PEAK_HALF:nbefore + PEAK_HALF, :]
        mad[on_g] = np.max(np.abs(peak) / nz[None, None, :], axis=(1, 2))
    return dict(rF_u=rF_u, rA_u=rA_u, rF_best=rF_best, rA_best=rA_best, rF_arg=rF_arg, rA_arg=rA_arg, mad=mad)


def matched_rstar(r, s, ci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, target):
    for rr in R_GRID:
        keep = r >= rr
        sc = score_kept_spikes(s[keep], ci[keep], ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)
        if np.isfinite(sc["median_rp"]) and sc["median_rp"] <= target:
            return float(rr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-h", type=float, default=27.0)
    ap.add_argument("--units", type=int, default=2)
    ap.add_argument("--n-ex", type=int, default=8)
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
    sdir = WV / "tight_reassign" / f"w{int(args.window_h)}h_ref"
    shutil.rmtree(sdir, ignore_errors=True)
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore = bank.nbefore
    ids = np.asarray(bank.unit_ids)
    a = nbefore + int(round(ASYM_MS[0] * FS / 1000.0))
    b = nbefore + int(round(ASYM_MS[1] * FS / 1000.0))
    units_per_tet = {int(g): int((ug == g).sum()) for g in np.unique(ug)}
    peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
    peak_by_tet = {int(gg): np.sort(peak_s[peak_g == gg]) for gg in np.unique(peak_g)}

    _, pool = run_matching(win, bank, method="wobble", n_jobs=N_JOBS,
                           method_kwargs=wobble_method_kwargs(bank, threshold=args.pool_factor * tsq_median(bank)))
    ps = pool["sample_index"].astype(np.int64)
    pci = pool["cluster_index"].astype(np.int64)
    C = all_template_cosines(win, bank, pool, a, b, S_MAX)
    rstar_full = matched_rstar(C["rF_u"], ps, pci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, args.rp_target)
    rstar_asym = matched_rstar(C["rA_u"], ps, pci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, args.rp_target)
    print(f"window @ {args.window_h:.0f}h: pool {pool.size:,}; asym samples [{a},{b}], shift +/-{S_MAX}", flush=True)
    print(f"matched r* (median rp<={args.rp_target}): full|0 = {rstar_full} ; asym|+-{S_MAX} = {rstar_asym}",
          flush=True)
    if rstar_full is None or rstar_asym is None:
        print("  precision target unreachable; abort", flush=True)
        return

    finite = np.isfinite(C["rF_u"]) & np.isfinite(C["rA_u"]) & np.isfinite(C["rA_best"]) & np.isfinite(C["mad"])
    dropped = finite & (C["rF_u"] >= rstar_full) & (C["rA_u"] < rstar_asym)
    kept = finite & (C["rF_u"] >= rstar_full) & (C["rA_u"] >= rstar_asym)
    counts = np.bincount(pci[finite], minlength=ids.size)
    picked = [u for u in np.argsort(counts)[::-1][:args.units] if counts[u] > 0]

    def summarize(name, mask, u_idx=None):
        m = mask if u_idx is None else (mask & (pci == u_idx))
        idx = np.flatnonzero(m)
        if idx.size == 0:
            print(f"  {name}: n=0", flush=True)
            return
        u_is_best = C["rA_arg"][idx] == (pci[idx] if u_idx is None else u_idx)
        nb = ~u_is_best
        nb_clears = nb & (C["rA_best"][idx] >= rstar_asym)
        print(f"  {name}: n={idx.size:>6} | assigned-u IS best tight-match {100*u_is_best.mean():5.1f}% | "
              f"NEIGHBOR wins tight {100*nb.mean():5.1f}% (of those, neighbor clears r*_asym "
              f"{100*nb_clears.sum()/max(nb.sum(),1):4.1f}%) | median rA_best {np.median(C['rA_best'][idx]):.2f} "
              f"rA_u {np.median(C['rA_u'][idx]):.2f}", flush=True)

    print("\n=== reassignment test: do tight-dropped spikes match a NEIGHBOR better than the assigned unit? ===",
          flush=True)
    print(f"units/tetrode: {units_per_tet}", flush=True)
    print("POOLED over all units:", flush=True)
    summarize("dropped-by-tightening", dropped)
    summarize("kept-by-both (control)", kept)
    for u_idx in picked:
        g = int(ug[u_idx])
        print(f"\nunit {ids[u_idx]} (tet {g}, {units_per_tet[g]} units on tetrode):", flush=True)
        summarize("dropped", dropped, u_idx)
        summarize("kept   ", kept, u_idx)

        # example figure: neighbor-wins dropped spikes -> u template (gray dashed) vs winner neighbor (red dotted)
        chans = np.flatnonzero(rec_groups == g)
        sel = np.flatnonzero(dropped & (pci == u_idx) & (C["rA_arg"] != u_idx))
        if sel.size == 0:
            print("    (no neighbor-wins dropped spikes to plot)", flush=True)
            continue
        pick = rng.choice(sel, size=min(args.n_ex, sel.size), replace=False)
        ne = pick.size
        fig, axes = plt.subplots(1, ne, figsize=(2.1 * ne, 2.6), squeeze=False)
        for c in range(ne):
            j = int(pick[c])
            v = int(C["rA_arg"][j])
            sample = int(ps[j])
            tr = np.asarray(win.get_traces(start_frame=sample - nbefore, end_frame=sample - nbefore + dense.shape[1],
                                           channel_ids=list(chan_ids[chans])), dtype=np.float32)
            tu = dense[u_idx][:, chans]
            tv = dense[v][:, chans]
            ymax = max(np.abs(tr).max(), np.abs(tu).max(), np.abs(tv).max(), 1.0)
            offv = 1.3 * ymax
            ax = axes[0][c]
            ax.axvspan(a, b, color="#cfe8ff", alpha=0.6, zorder=0)
            for ch in range(len(chans)):
                ax.plot(tr[:, ch] - ch * offv, color="0.1", lw=0.9)
                ax.plot(tu[:, ch] - ch * offv, color="0.55", lw=0.9, ls="--")
                ax.plot(tv[:, ch] - ch * offv, color="#c0392b", lw=0.9, ls=":")
            ax.axvline(nbefore, color="0.7", lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-3.4 * offv, 1.4 * offv)
            ax.set_title(f"{C['mad'][j]:.1f}MAD\nrA_u{C['rA_u'][j]:.2f} rA_v{C['rA_best'][j]:.2f}\n->u{ids[v]}",
                         fontsize=6.5)
        fig.suptitle(f"u{ids[u_idx]} tet {g}: tight-DROPPED spikes where a NEIGHBOR wins the trough window  "
                     f"(black=snippet, gray--=u{ids[u_idx]} template, red:=winner-neighbor template; shaded=asym)",
                     fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        p = WV / f"tight_reassign_u{ids[u_idx]}_w{int(args.window_h)}h.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"    wrote {p.name} ({sel.size} neighbor-wins dropped spikes total)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

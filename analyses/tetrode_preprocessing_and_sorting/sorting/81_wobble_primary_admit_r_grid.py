"""Phase A refinement: the (wobble admit x cosine r) 2D grid -- do the two knobs interact, or does r govern?

The bake-off (79) swept r at a FIXED permissive admit (0.45x); the admit value itself was an unvalidated
heuristic. The two knobs gate DIFFERENT axes -- the wobble admit threshold ~ an amplitude gate
(objective 2*conv-||t||^2 = ||t||^2*(2a-1), so factor f ~ a>=(1+f)/2), while the cosine r is a
scale-invariant SHAPE gate -- linked by a = r*||snip||/||t|| but not interchangeable. This maps the full
{admit} x {r} plane to answer three coupled questions before Phase B locks both in:
  1. SATURATION: as the admit drops, does the cosine-gated output stop changing (admit non-binding) -- so
     0.45x was safe -- or keep recovering spikes (admit was binding, capping what r can reach)?
  2. INTERACTION: is the achievable (rp, >=10 cov) frontier governed by r alone (cov-vs-r lines overlap
     across admits), or is there an off-(0.45x) (admit, r) corner that does better?
  3. B_w CONTROL: at each admit, the same wobble run WITHOUT the shape gate (r off) is wobble ALONE -- the
     validation ladder's matcher-swap control. Its median rp vs circus[0.8,inf] locks the B_w full-48h
     factor: the admit whose no-gate rp matches circus (~0.55-0.60x by the a>=(1+f)/2 <-> circus a>=0.8 map).
r is a FREE post-hoc filter, so each admit = one wobble run yields its whole r-curve AND its no-gate point --
the 2D grid + B_w lock costs the same as a 1D admit sweep. Uses the analyzer-free scorer
(_wobble_eval.score_kept_spikes).

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/81_wobble_primary_admit_r_grid.py \
        [--windows-h 11 19 27 40] [--admit-factors 0.25 0.35 0.45 0.55 0.65] [--r-gates 0.50 0.55 0.60 0.65 0.70 0.75]
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
                        per_spike_cosine, run_matching, tsq_median, wobble_method_kwargs)
from _wobble_eval import detect_window_peaks, score_kept_spikes
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}
OP_R = 0.60   # the operating r, for the saturation (count vs admit) panel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", type=float, nargs="+", default=[11.0, 19.0, 27.0, 40.0],
                    help="window starts (h); 4 by default -- the 2D grid is 5x the runs of a 1D r-sweep, and "
                    "6-window r-generalization is already shown by the bake-off (script 79)")
    ap.add_argument("--admit-factors", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55, 0.65],
                    help="wobble admit thresholds (x tsq_median); each is one wobble run")
    ap.add_argument("--r-gates", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                    help="cosine shape-gate values, post-filtered FREE on each admit run")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    admits, r_gates = sorted(args.admit_factors), sorted(args.r_gates)
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    # results[h] = {"circus": row, "grid": {admit: {r: row}}}
    results = {}
    for h in args.windows_h:
        a0 = int(h * 3600 * FS)
        b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a0, b0)
        win.reset_times()
        nfr = win.get_num_frames()
        sdir = WV / "admit_r_grid" / f"w{int(h)}h_ref"
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.perf_counter()
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=args.n_jobs, seed=0)
        ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
        med = tsq_median(bank)
        peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=args.n_jobs)
        peak_by_tet = {int(gg): np.sort(peak_s[peak_g == gg]) for gg in np.unique(peak_g)}
        print(f"\n=== @ {h:.0f}h: {bank.unit_ids.size} units, {(amp_mad >= 10).sum():,} >=10 MAD events, "
              f"setup {time.perf_counter()-t0:.0f}s ===", flush=True)

        _, cs = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=args.n_jobs)
        circ = score_kept_spikes(cs["sample_index"].astype(np.int64), cs["cluster_index"].astype(np.int64),
                                 ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)
        print(f"  circus[0.8,inf]: rp {circ['median_rp']:.4f} | >=10 {circ['cov10']:.1f}% | n {circ['n_spikes']:,}",
              flush=True)

        grid = {}
        nogate = {}
        for f in admits:
            tw = time.perf_counter()
            _, sp = run_matching(win, bank, method="wobble", n_jobs=args.n_jobs,
                                 method_kwargs=wobble_method_kwargs(bank, threshold=f * med))
            r_all = per_spike_cosine(sp, bank, win)
            s = sp["sample_index"].astype(np.int64)
            ci = sp["cluster_index"].astype(np.int64)
            # no-gate = wobble ALONE (r filter OFF): the B_w matcher-swap control's operating point per admit
            nogate[f] = score_kept_spikes(s, ci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr)
            grid[f] = {}
            for rg in r_gates:
                keep = r_all >= rg
                grid[f][rg] = score_kept_spikes(s[keep], ci[keep], ug, peak_s, peak_g, amp_mad,
                                                peak_by_tet, nfr)
            line = " ".join(f"r{rg:.2f}:rp{grid[f][rg]['median_rp']:.3f}/c{grid[f][rg]['cov10']:.0f}"
                            for rg in r_gates)
            print(f"  admit {f:.2f}x ({sp.size:,}sp, {time.perf_counter()-tw:.0f}s): "
                  f"nogate:rp{nogate[f]['median_rp']:.3f}/c{nogate[f]['cov10']:.0f} {line}", flush=True)
        results[f"{h:.0f}h"] = {
            "circus": circ,
            "nogate": {f"{f:.2f}": nogate[f] for f in admits},
            "grid": {f"{f:.2f}": {f"{rg:.2f}": grid[f][rg] for rg in r_gates} for f in admits}}

    _summarize_and_plot(results, admits, r_gates)


def _agg(results, admits, r_gates, field):
    """Median across windows of `field` for each (admit, r) cell -> 2D array [admit, r]."""
    hs = list(results)
    out = np.full((len(admits), len(r_gates)), np.nan)
    for i, f in enumerate(admits):
        for j, rg in enumerate(r_gates):
            vals = [results[h]["grid"][f"{f:.2f}"][f"{rg:.2f}"][field] for h in hs]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                out[i, j] = float(np.median(vals))
    return out


def _agg_nogate(results, admits, field):
    """Median across windows of `field` for wobble ALONE (no shape gate) at each admit -> 1D [admit]."""
    hs = list(results)
    out = np.full(len(admits), np.nan)
    for i, f in enumerate(admits):
        vals = [results[h]["nogate"][f"{f:.2f}"][field] for h in hs]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[i] = float(np.median(vals))
    return out


def _summarize_and_plot(results, admits, r_gates):
    hs = list(results)
    rp = _agg(results, admits, r_gates, "median_rp")
    cov = _agg(results, admits, r_gates, "cov10")
    nsp = _agg(results, admits, r_gates, "n_spikes")
    circ_rp = float(np.median([results[h]["circus"]["median_rp"] for h in hs]))
    circ_cov = float(np.median([results[h]["circus"]["cov10"] for h in hs]))
    ng_rp = _agg_nogate(results, admits, "median_rp")
    ng_cov = _agg_nogate(results, admits, "cov10")
    bw_i = int(np.nanargmin(np.abs(ng_rp - circ_rp)))

    print("\n=== wobble ALONE (no shape gate) vs circus -- locks the B_w matcher-swap control factor ===",
          flush=True)
    print(f"  circus[0.8,inf]: rp {circ_rp:.4f} | cov10 {circ_cov:.1f}%", flush=True)
    print("  admit    nogate_rp  cov10   |rp-circ|", flush=True)
    for i, f in enumerate(admits):
        print(f"  {f:.2f}x    {ng_rp[i]:.4f}    {ng_cov[i]:5.1f}    {abs(ng_rp[i] - circ_rp):.4f}",
              flush=True)
    print(f"  -> B_w factor (no-gate rp closest to circus): {admits[bw_i]:.2f}x "
          f"(rp {ng_rp[bw_i]:.4f} vs circus {circ_rp:.4f}, cov10 {ng_cov[bw_i]:.1f}%)", flush=True)

    print("\n=== aggregated across windows (median): cov10 [admit x r] ===", flush=True)
    print("        " + "  ".join(f"r{rg:.2f}" for rg in r_gates), flush=True)
    for i, f in enumerate(admits):
        print(f"  {f:.2f}x  " + "  ".join(f"{cov[i, j]:5.1f}" for j in range(len(r_gates))), flush=True)
    print("=== median_rp [admit x r] (0.000 = fully clean) ===", flush=True)
    for i, f in enumerate(admits):
        print(f"  {f:.2f}x  " + "  ".join(f"{rp[i, j]:.3f}" for j in range(len(r_gates))), flush=True)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(admits)))
    # (a) cov10 vs r, one line per admit -> overlap = admit non-binding (r governs); separation = interaction
    for i, f in enumerate(admits):
        ax[0].plot(r_gates, cov[i], "o-", color=colors[i], label=f"admit {f:.2f}x")
    ax[0].axhline(circ_cov, color="0.4", ls="--", label="circus")
    ax[0].set_xlabel("cosine gate r")
    ax[0].set_ylabel(">=10 MAD coverage % (median over windows)")
    ax[0].set_title("Coverage vs r per admit\n(lines overlap => admit non-binding)")
    ax[0].legend(fontsize=8)
    # (b) median rp vs r, one line per admit
    for i, f in enumerate(admits):
        ax[1].plot(r_gates, rp[i], "o-", color=colors[i], label=f"admit {f:.2f}x")
    ax[1].axhline(0.1, color="#c0392b", ls=":", label="BombCell 0.1")
    ax[1].axhline(circ_rp, color="0.4", ls="--", label="circus")
    ax[1].set_xlabel("cosine gate r")
    ax[1].set_ylabel("median rp_contamination")
    ax[1].set_title("Precision vs r per admit")
    ax[1].legend(fontsize=8)
    # (c) SATURATION: kept-spike count vs admit at the operating r -> flat = non-binding
    jr = int(np.argmin([abs(rg - OP_R) for rg in r_gates]))
    ax[2].plot(admits, nsp[:, jr] / 1e6, "s-", color="#3b7dd8")
    ax[2].set_xlabel("admit factor (x tsq_median)")
    ax[2].set_ylabel(f"kept spikes (millions) at r>={r_gates[jr]:.2f}")
    ax[2].set_title("Saturation: recovery vs admit\n(flat at low admit => 0.45x was non-binding)")
    # (d) B_w control: wobble ALONE (no gate) rp & cov vs admit, with circus reference -> the matched factor
    l1 = ax[3].plot(admits, ng_rp, "o-", color="#c0392b", label="wobble-alone rp")
    ax[3].axhline(circ_rp, color="#c0392b", ls="--", lw=1, label="circus rp")
    ax[3].axvline(admits[bw_i], color="0.5", ls=":", lw=1)
    ax[3].set_xlabel("admit factor (x tsq_median)")
    ax[3].set_ylabel("median rp_contamination", color="#c0392b")
    ax[3].tick_params(axis="y", labelcolor="#c0392b")
    ax[3].set_title("B_w control: wobble alone vs circus\n(match rp -> B_w factor)")
    ax3b = ax[3].twinx()
    l2 = ax3b.plot(admits, ng_cov, "s-", color="#2e7d32", label="wobble-alone cov10")
    ax3b.axhline(circ_cov, color="#2e7d32", ls="--", lw=1, label="circus cov10")
    ax3b.set_ylabel(">=10 MAD coverage %", color="#2e7d32")
    ax3b.tick_params(axis="y", labelcolor="#2e7d32")
    lines = l1 + l2
    ax[3].legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")
    fig.suptitle("Wobble admit x cosine-r 2D grid: does the admit interact with r, or does r govern the frontier?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figp = WV / "admit_r_grid.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)
    (WV / "admit_r_grid.json").write_text(json.dumps(
        {"admits": admits, "r_gates": r_gates, "op_r": OP_R, "results": results}, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / 'admit_r_grid.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

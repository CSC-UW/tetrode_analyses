"""Prototype a SCALE-INVARIANT per-unit gate for wobble and test whether ONE fixed fraction generalizes
across windows -- the generalization the absolute threshold lacked (0.55x median under-covered at h=40).

Wobble's absolute threshold gates `2*conv - ||t||^2` (amplitude^2), which conflates fit-quality with
template energy ||t||^2 (varies 1e6-1e7 across units) -> a single global value is an inconsistent relative
bar. The fix is a scale-invariant per-unit gate. We prototype it as a POST-FILTER on a generously-admitted
wobble run: for each accepted spike compute, against its ASSIGNED unit's template,
    a = conv / ||t||^2          (amplitude scaling; circus-omp gates exactly this, a>=0.8)
    r = conv / (||t|| * ||snip||)   (cosine / shape match; scale-invariant, preserves sub-threshold spikes)
then sweep a fixed gate (a* or r*) and recompute rp_contamination (Llobet, SI internals) + >=12 MAD
coverage per window. If a fixed fraction clusters the 3 windows' (rp, coverage) tightly, the gate
generalizes (unlike the absolute threshold). NOTE the amplitude gate `a` rejects low-amplitude spikes
(throws away wobble's sub-threshold-recovery edge); the cosine gate `r` keeps them -> both are reported.

Pre-dedup wobble output (cluster_index -> assigned unit -> template). Generous admit = 0.45x ||t||^2 median
per window (so the post-filter, not wobble's own threshold, is the binding constraint).

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes \
        python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/74_wobble_normgate_prototype.py
"""
import json
import pathlib
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.metrics.quality.misc_metrics import (_compute_rp_contamination_one_unit,
                                                         _compute_rp_violations_numba)

from _mp_common import (_unit_groups_from_mask, build_templates_object, materialize_span,
                        run_matching, wobble_method_kwargs)
from _wobble_eval import _within_tol, detect_window_peaks, tsq_median
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
WINDOW_STARTS_H = [5.0, 26.0, 40.0]
N_JOBS = 16
TOL = int(0.5e-3 * FS)
ADMIT_FACTOR = 0.45                 # generous wobble admit (x ||t||^2 median) so the post-gate is binding
T_R = int(round(1.0 * FS * 1e-3))   # 1.0 ms refractory (SI rp_violation default), t_c = 0
A_GATES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
R_GATES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MIN_SPK = 50


def rp_contam(samples, total_samples):
    """Llobet rp_contamination for one spike train (SI internals; t_c=0, t_r=1ms)."""
    if samples.size < 2:
        return np.nan
    nv = _compute_rp_violations_numba(np.sort(samples).astype(np.int64), 0, T_R)
    return _compute_rp_contamination_one_unit(nv, int(samples.size), int(total_samples), 0, T_R)


def median_rp(kept_by_unit, nfr):
    vals = [rp_contam(s, nfr) for s in kept_by_unit.values() if s.size >= MIN_SPK]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def cov12(kept_by_tet, peak_s, peak_g, amp_mad):
    claimed = _within_tol(peak_s, peak_g, kept_by_tet, TOL)
    big = amp_mad >= 12
    return float(claimed[big].mean() * 100) if big.any() else float("nan")


def main():
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)
    results = {}
    for h in WINDOW_STARTS_H:
        a0 = int(h * 3600 * FS)
        b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a0, b0)
        win.reset_times()
        nfr = win.get_num_frames()
        sdir = WV / "normgate" / f"w{int(h)}h_ref"
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.perf_counter()
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
        ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)  # per cluster_index
        dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
        nbefore_t, n_samp_t = bank.nbefore, dense.shape[1]
        tsq_u = np.array([float((dense[i][:, np.flatnonzero(rec_groups == ug[i])] ** 2).sum())
                          for i in range(dense.shape[0])])
        med = tsq_median(bank)
        peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
        thr = ADMIT_FACTOR * med
        _, spikes = run_matching(win, bank, method="wobble",
                                 method_kwargs=wobble_method_kwargs(bank, threshold=thr), n_jobs=N_JOBS)
        s_all = spikes["sample_index"].astype(np.int64)
        ci_all = spikes["cluster_index"].astype(np.int64)
        g_all = ug[ci_all]
        print(f"=== @ {h:.0f}h: {bank.unit_ids.size} units, admit thr={thr:.4g} -> {s_all.size:,} spikes, "
              f"{(amp_mad>=12).sum():,} >=12 MAD events; setup {time.perf_counter()-t0:.0f}s ===", flush=True)

        # per-spike fit quality a, r against the ASSIGNED unit's template (batched per tetrode)
        a_all = np.full(s_all.size, np.nan)
        r_all = np.full(s_all.size, np.nan)
        off_all = s_all - nbefore_t
        valid_all = (off_all >= 0) & (off_all + n_samp_t <= nfr)
        for g in np.unique(g_all):
            chans = np.flatnonzero(rec_groups == g)
            tr = np.asarray(win.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)  # (nfr,4)
            on_g = np.flatnonzero((g_all == g) & valid_all)
            if on_g.size == 0:
                continue
            cols = np.arange(n_samp_t)
            for u_idx in np.unique(ci_all[on_g]):
                sel = on_g[ci_all[on_g] == u_idx]
                offs = off_all[sel]
                snips = tr[offs[:, None] + cols[None, :], :]               # (n_sel, n_samp_t, 4)
                templ = dense[u_idx][:, chans]                             # (n_samp_t, 4)
                conv = np.einsum("ntc,tc->n", snips, templ)
                snip_sq = np.einsum("ntc,ntc->n", snips, snips)
                a_all[sel] = conv / tsq_u[u_idx]
                denom = np.sqrt(tsq_u[u_idx] * snip_sq)
                r_all[sel] = np.where(denom > 0, conv / denom, np.nan)
        fin = valid_all & np.isfinite(a_all) & np.isfinite(r_all)
        s_f, g_f, ci_f, a_f, r_f = s_all[fin], g_all[fin], ci_all[fin], a_all[fin], r_all[fin]

        def sweep(qual, gates):
            rows = []
            for q in gates:
                keep = qual >= q
                sk, gk, ck = s_f[keep], g_f[keep], ci_f[keep]
                by_unit, by_tet = {}, {}
                for u_idx in np.unique(ck):
                    by_unit[int(u_idx)] = np.sort(sk[ck == u_idx])
                for gg in np.unique(gk):
                    by_tet[int(gg)] = np.sort(sk[gk == gg])
                rows.append({"gate": q, "retained_frac": float(keep.mean()),
                             "median_rp": median_rp(by_unit, nfr),
                             "cov12": cov12(by_tet, peak_s, peak_g, amp_mad)})
            return rows

        a_rows = sweep(a_f, A_GATES)
        r_rows = sweep(r_f, R_GATES)
        for tag, rows in (("a", a_rows), ("r", r_rows)):
            for rw in rows:
                print(f"  {tag}>={rw['gate']:.2f}: keep {rw['retained_frac']*100:4.0f}% | med_rp "
                      f"{rw['median_rp']:.4f} | >=12 cov {rw['cov12']:.1f}%", flush=True)
        results[f"{h:.0f}h"] = {"tsq_median": med, "admit_thr": thr, "n_admitted": int(s_all.size),
                                "amplitude_gate": a_rows, "cosine_gate": r_rows}

    # ---- generalization: spread of (rp, cov) across windows at each fixed gate ----
    hs = list(results)

    def spread(kind, gates, field):
        out = {}
        for i, q in enumerate(gates):
            vals = [results[h][kind][i][field] for h in hs]
            v = np.array([x for x in vals if np.isfinite(x)])
            out[q] = (float(v.min()), float(v.max()), float(v.max() - v.min()) if v.size else float("nan"))
        return out

    print("\n=== generalization (range across 5/26/40h at each fixed gate) ===", flush=True)
    for kind, gates, label in (("amplitude_gate", A_GATES, "a"), ("cosine_gate", R_GATES, "r")):
        cov_sp = spread(kind, gates, "cov12")
        rp_sp = spread(kind, gates, "median_rp")
        for q in gates:
            print(f"  {label}>={q:.2f}: >=12cov range {cov_sp[q][0]:.1f}-{cov_sp[q][1]:.1f}% "
                  f"(spread {cov_sp[q][2]:.1f}) | rp range {rp_sp[q][0]:.3f}-{rp_sp[q][1]:.3f}", flush=True)
    print("  (compare: absolute 0.55x threshold gave >=12cov 82.7-99.4% = 16.7 pt spread)", flush=True)

    # ---- figure: cov & rp vs gate, one line per window, for amplitude and cosine gates ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"5h": "#2e8b57", "26h": "#3b7dd8", "40h": "#d8743b"}
    for col, (kind, gates, label) in enumerate((("amplitude_gate", A_GATES, "a = conv/||t||^2"),
                                                ("cosine_gate", R_GATES, "r = cos(snippet, template)"))):
        for h in hs:
            rows = results[h][kind]
            ax[0][col].plot(gates, [rw["cov12"] for rw in rows], "o-", color=colors[h], label=h)
            ax[1][col].plot(gates, [rw["median_rp"] for rw in rows], "o-", color=colors[h], label=h)
        ax[0][col].set_title(f"gate {label}: >=12 MAD coverage")
        ax[0][col].set_ylabel("% >=12 MAD claimed")
        ax[0][col].set_ylim(0, 102)
        ax[0][col].legend(fontsize=8)
        ax[1][col].axhline(0.1, color="#c0392b", ls="--", label="BombCell rp 0.1")
        ax[1][col].set_title(f"gate {label}: median rp_contamination")
        ax[1][col].set_ylabel("median rp_contamination")
        ax[1][col].set_xlabel("gate value")
        ax[1][col].legend(fontsize=8)
    fig.suptitle("Scale-invariant per-unit gate on wobble: does one fixed fraction generalize across windows?\n"
                 "(tight clustering of the 3 window lines = generalizes, unlike the absolute threshold)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figp = WV / "normgate_generalization.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)

    (WV / "normgate.json").write_text(json.dumps(
        {"admit_factor": ADMIT_FACTOR, "a_gates": A_GATES, "r_gates": R_GATES, "results": results}, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / 'normgate.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

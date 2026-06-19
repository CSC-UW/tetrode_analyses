"""De-risk step 3: full wobble-vs-circus head-to-head on the three task-1 windows, at MATCHED precision.

Applies the calibrated wobble threshold T_h (from 70_wobble_threshold_calib.py @ h=26) to each task-1
window (5/26/40 h). For each window, ONE shared MS5 bank feeds BOTH matchers; each is deduped and scored
on all four axes:
  1. >=12 MAD coverage (detection completeness)
  2. precision: median rp_contamination + per-tier well-isolated unit counts
  3. MS5 agreement: per-tier match_frac + mean Jaccard vs the bank's source MS5 sort
  4. over-detection: spurious fraction (matcher spikes with no detected peak) + duplicate fractions

Also checks GENERALIZATION: did wobble's precision at T_h stay <= circus-omp's in each window (the matched
point was chosen on h=26)? Writes wobble_vs_circus/headtohead.json + coverage_bands.png + precision_agreement.png.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/71_wobble_vs_circus.py
"""
import json
import pathlib
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _mp_common import (_unit_groups_from_mask, build_templates_object, dedup_sorting,
                        materialize_span, run_matching, wobble_method_kwargs)
from _track_eval import score_reconstruction
from _wobble_eval import (AMP_BINS, coverage_by_band, detect_window_peaks, over_detection,
                          precision_summary, quality_df, spurious_fraction, tsq_median)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
WINDOW_STARTS_H = [5.0, 26.0, 40.0]
N_JOBS = 16
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}
BAND_LABELS = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
               for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:])]


def score_arm(mp, spikes, win, ug_map, peaks, ms5, ms5_qm):
    """Set group -> dedup -> all four axes for one matcher arm. Returns metrics dict."""
    peak_s, peak_g, amp_mad = peaks
    mp.set_property("group", np.array([ug_map[int(u)] for u in mp.unit_ids]))
    dd = dedup_sorting(mp, win)
    qm = quality_df(dd, win, n_jobs=N_JOBS)
    cov, _ = coverage_by_band(dd, peak_s, peak_g, amp_mad)
    agree = score_reconstruction(ms5, ms5_qm, dd)
    return {
        "n_spikes_raw": int(spikes.size),
        "n_units_deduped": int(dd.get_num_units()),
        "coverage": cov,
        "precision": precision_summary(dd, qm),
        "spurious_frac": spurious_fraction(dd, peak_s, peak_g),
        "over_detection": over_detection(dd, int(spikes.size)),
        "ms5_agreement": agree,
    }


def main():
    calib = json.loads((WV / "threshold_calib_w26h.json").read_text())
    factor = calib.get("chosen_factor")
    if factor is None:
        raise SystemExit("threshold_calib_w26h.json has chosen_factor=null: wobble could not match "
                         "circus-omp precision on h=26. Re-run/extend calibration before the head-to-head.")
    # Apply the calibrated FACTOR per-window (threshold = factor x that window's ||t||^2 median): the
    # objective scales with ||t||^2, so a factor generalizes across windows better than a fixed absolute
    # threshold. Per-window rp is reported so we can SEE whether the matched point held (generalization).
    print(f"using calibrated wobble factor = {factor}x median ||t||^2 (T_h@26h={calib.get('chosen_T_h'):.4g})\n",
          flush=True)

    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))

    results = {}
    for h in WINDOW_STARTS_H:
        a = int(h * 3600 * FS)
        b = min(a + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a, b)
        win.reset_times()
        sdir = WV / f"w{int(h)}h" / "ref_sort"
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.perf_counter()
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=True, n_jobs=N_JOBS, seed=0)
        ug_map = {int(u): int(g) for u, g in
                  zip(bank.unit_ids, _unit_groups_from_mask(bank.sparsity.mask, rec_groups))}
        ms5_qm = quality_df(ms5, win, n_jobs=N_JOBS)  # for per-tier reference unit selection
        peaks = detect_window_peaks(win, n_jobs=N_JOBS)
        med_h = tsq_median(bank)
        thr_h = factor * med_h
        print(f"=== window @ {h:.0f}h: {bank.unit_ids.size} bank units, {peaks[0].size:,} events "
              f"({(peaks[2]>=12).sum():,} >=12 MAD); wobble thr={thr_h:.4g} ({factor}x med {med_h:.3g}); "
              f"setup {time.perf_counter()-t0:.0f}s ===", flush=True)

        mp_c, sp_c = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=N_JOBS)
        m_c = score_arm(mp_c, sp_c, win, ug_map, peaks, ms5, ms5_qm)
        mp_w, sp_w = run_matching(win, bank, method="wobble",
                                  method_kwargs=wobble_method_kwargs(bank, threshold=thr_h), n_jobs=N_JOBS)
        m_w = score_arm(mp_w, sp_w, win, ug_map, peaks, ms5, ms5_qm)

        rp_c = m_c["precision"]["median_rp_contamination"]
        rp_w = m_w["precision"]["median_rp_contamination"]
        matched = bool(np.isfinite(rp_w) and np.isfinite(rp_c) and rp_w <= rp_c)
        print(f"  circus : {m_c['n_spikes_raw']:>8,}sp {m_c['n_units_deduped']:>3}u | >=12MAD "
              f"{m_c['coverage']['>=12_pooled']:5.1f}% | rp {rp_c:.4f} | spur {m_c['spurious_frac']*100:4.1f}% "
              f"| MS5 mod match {m_c['ms5_agreement']['moderate']['match_frac']}", flush=True)
        print(f"  wobble : {m_w['n_spikes_raw']:>8,}sp {m_w['n_units_deduped']:>3}u | >=12MAD "
              f"{m_w['coverage']['>=12_pooled']:5.1f}% | rp {rp_w:.4f} | spur {m_w['spurious_frac']*100:4.1f}% "
              f"| MS5 mod match {m_w['ms5_agreement']['moderate']['match_frac']}", flush=True)
        print(f"  precision matched at T_h (wobble rp <= circus rp): {matched}", flush=True)
        results[f"{h:.0f}h"] = {"circus": m_c, "wobble": m_w, "precision_matched": matched,
                                "wobble_threshold": float(thr_h), "tsq_median": float(med_h)}

    # ---------- figures ----------
    hs = list(results)
    # coverage by band, per window
    fig, axes = plt.subplots(1, len(hs), figsize=(5 * len(hs), 4.4), squeeze=False)
    centers = [lo if not np.isfinite(hi) else 0.5 * (lo + hi)
               for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:])]
    for j, hk in enumerate(hs):
        ax = axes[0][j]
        for arm, col in (("circus", "#d8743b"), ("wobble", "#3b7dd8")):
            cov = results[hk][arm]["coverage"]
            ax.plot(centers, [cov[b] for b in BAND_LABELS], "o-", color=col, label=arm)
        ax.set_xlabel("event amplitude (MAD)")
        ax.set_ylabel("% events claimed")
        ax.set_ylim(0, 100)
        ax.axhline(90, color="0.8", ls=":")
        ax.set_title(f"@ {hk}: coverage vs amplitude")
        ax.legend(fontsize=8)
    fig.suptitle("Coverage by amplitude band: wobble vs circus-omp (at matched precision T_h)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(WV / "coverage_bands.png", dpi=130)
    plt.close(fig)

    # precision + coverage + MS5 agreement grouped bars (one panel each; arm = circus/wobble)
    x = np.arange(len(hs))
    w = 0.38
    panels = [
        ("median rp_contamination", "Precision (lower=better)", None,
         lambda m: m["precision"]["median_rp_contamination"]),
        ("% >=12 MAD events claimed", ">=12 MAD coverage (higher=better)", (0, 100),
         lambda m: m["coverage"][">=12_pooled"]),
        ("MS5 match_frac (moderate tier)", "MS5 agreement (higher=better)", (0, 1),
         lambda m: m["ms5_agreement"]["moderate"]["match_frac"]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for axi, (ylab, title, ylim, get) in zip(ax, panels):
        axi.bar(x - w / 2, [get(results[h]["circus"]) for h in hs], w, color="#d8743b", label="circus")
        axi.bar(x + w / 2, [get(results[h]["wobble"]) for h in hs], w, color="#3b7dd8", label="wobble")
        axi.set_xticks(x)
        axi.set_xticklabels(hs)
        axi.set_ylabel(ylab)
        axi.set_title(title)
        if ylim is not None:
            axi.set_ylim(*ylim)
        axi.legend(fontsize=8)
    fig.suptitle("Wobble vs circus-omp: precision, coverage, MS5 agreement across task-1 windows", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(WV / "precision_agreement.png", dpi=130)
    plt.close(fig)

    out = {"chosen_factor": factor, "calib_T_h_at_26h": calib.get("chosen_T_h"), "win_s": WIN_S,
           "windows_h": WINDOW_STARTS_H, "circus_method_kwargs": {"amplitudes": [0.8, None]},
           "results": results}
    (WV / "headtohead.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {WV / 'coverage_bands.png'}\nwrote {WV / 'precision_agreement.png'}\n"
          f"wrote {WV / 'headtohead.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

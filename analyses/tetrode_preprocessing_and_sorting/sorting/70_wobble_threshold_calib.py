"""De-risk step 2: calibrate wobble's raw-unit detection threshold to MATCHED PRECISION vs circus-omp.

Wobble's `threshold` is on the normalized objective (amplitude^2, raw units) -- the smoke run (script 69)
showed the meaningful scale is ~the per-template ||t||^2 distribution (median ~3e6) and that wobble
over-produces at that scale. To compare matchers FAIRLY we must not hand wobble a coverage advantage by
running it looser. So:

  1. circus-omp (amplitudes=[0.8,inf], the production setting) -> dedup -> PRECISION TARGET
     = median rp_contamination over units with >=50 spikes (rate-gated, not contamination-gated).
  2. sweep wobble threshold over a ||t||^2-scaled grid; for each: dedup -> precision + >=12 MAD coverage
     + spurious fraction + over-detection.
  3. T_h = the SMALLEST threshold whose median rp_contamination <= circus-omp's (contamination is
     monotone-decreasing in threshold, so this is the boundary = max coverage at matched-or-better
     precision). If none qualify, wobble cannot match precision on this window -> reported as such.

Writes wobble_vs_circus/threshold_calib_w{h}h.json + threshold_calib_w{h}h.png. Default window h=26
(busiest/most stringent task-1 window). Pass an hour as argv[1] to calibrate another window.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/70_wobble_threshold_calib.py [26]
"""
import json
import pathlib
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _mp_common import (_unit_groups_from_mask, build_templates_object, dedup_sorting,
                        materialize_span, run_matching, wobble_method_kwargs)
from _wobble_eval import (coverage_by_band, detect_window_peaks, over_detection, precision_summary,
                          quality_df, spurious_fraction, tsq_median)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}
# x median ||t||^2; LOW factor = looser/more spikes/higher rp_contamination. The smoke run + a first
# sweep showed circus-omp[0.8,inf] (~1.35M spikes, rp~0.156) sits BELOW 1x median for wobble (wobble is
# already over-precise at 1x: rp~0, fewer spikes), so the matched-precision crossing is at <1x median.
# This grid brackets that crossing (0.4x over-loose -> 1.3x over-precise) to find wobble's fair coverage.
GRID_FACTORS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.3]


def main():
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
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
    med = tsq_median(bank)
    print(f"window @ {h:.0f}h: {bank.unit_ids.size} bank units, ||t||^2 median={med:.3g}, "
          f"setup {time.perf_counter()-t0:.0f}s", flush=True)

    print("detecting peaks (5.5 MAD locally-exclusive) ...", flush=True)
    peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
    print(f"  {peak_s.size:,} detected events; >=12 MAD: {(amp_mad>=12).sum():,}", flush=True)

    def score(mp, spikes):
        mp.set_property("group", np.array([ug_map[int(u)] for u in mp.unit_ids]))
        dd = dedup_sorting(mp, win)
        qm = quality_df(dd, win, n_jobs=N_JOBS)
        cov, _ = coverage_by_band(dd, peak_s, peak_g, amp_mad)
        return dd, {"precision": precision_summary(dd, qm), "coverage": cov,
                    "spurious_frac": spurious_fraction(dd, peak_s, peak_g),
                    "over_detection": over_detection(dd, int(spikes.size)),
                    "n_units_deduped": int(dd.get_num_units())}

    # --- circus-omp baseline (precision target) ---
    t0 = time.perf_counter()
    mp_c, sp_c = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=N_JOBS)
    _, m_c = score(mp_c, sp_c)
    target_rp = m_c["precision"]["median_rp_contamination"]
    print(f"\ncircus-omp [0.8,inf]: {sp_c.size:,} spikes -> dedup {m_c['n_units_deduped']}u | "
          f"median rp_contam={target_rp:.4f} | >=12MAD cov={m_c['coverage']['>=12_pooled']:.1f}% | "
          f"spurious={m_c['spurious_frac']*100:.1f}% | {time.perf_counter()-t0:.0f}s", flush=True)

    # --- wobble sweep ---
    grid = [round(f * med, 1) for f in GRID_FACTORS]
    wobble_runs = []
    print(f"\nwobble sweep over threshold grid {[f'{g:.3g}' for g in grid]}:", flush=True)
    for thr in grid:
        t0 = time.perf_counter()
        mp_w, sp_w = run_matching(win, bank, method="wobble",
                                  method_kwargs=wobble_method_kwargs(bank, threshold=thr), n_jobs=N_JOBS)
        _, m_w = score(mp_w, sp_w)
        rp = m_w["precision"]["median_rp_contamination"]
        cov12 = m_w["coverage"][">=12_pooled"]
        print(f"  thr={thr:>12.4g} ({thr/med:.1f}x): {sp_w.size:>8,} sp -> {m_w['n_units_deduped']:>3}u | "
              f"rp={rp:.4f} | >=12MAD cov={cov12:.1f}% | spur={m_w['spurious_frac']*100:.1f}% | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        wobble_runs.append({"threshold": thr, "factor": round(thr / med, 3), "n_spikes": int(sp_w.size),
                            **m_w})

    # --- pick T_h: smallest threshold whose median rp_contamination <= circus-omp's ---
    qualifying = [r for r in wobble_runs
                  if np.isfinite(r["precision"]["median_rp_contamination"])
                  and np.isfinite(target_rp)
                  and r["precision"]["median_rp_contamination"] <= target_rp]
    T_h = min(qualifying, key=lambda r: r["threshold"]) if qualifying else None
    if T_h is None:
        print(f"\n!! NO wobble threshold reached circus-omp precision ({target_rp:.4f}); "
              f"best wobble rp={min(r['precision']['median_rp_contamination'] for r in wobble_runs):.4f}. "
              f"Wobble cannot match precision on this window with this grid -- extend grid upward or "
              f"conclude circus-omp is preferred.", flush=True)
    else:
        print(f"\nchosen T_h = {T_h['threshold']:.4g} ({T_h['factor']:.1f}x median): "
              f"rp={T_h['precision']['median_rp_contamination']:.4f} <= circus {target_rp:.4f}, "
              f">=12MAD cov={T_h['coverage']['>=12_pooled']:.1f}% (circus {m_c['coverage']['>=12_pooled']:.1f}%)",
              flush=True)

    # --- figure: precision + coverage/spurious vs threshold ---
    thr_arr = np.array([r["threshold"] for r in wobble_runs])
    rp_arr = np.array([r["precision"]["median_rp_contamination"] for r in wobble_runs])
    cov_arr = np.array([r["coverage"][">=12_pooled"] for r in wobble_runs])
    spur_arr = np.array([r["spurious_frac"] * 100 for r in wobble_runs])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(thr_arr, rp_arr, "o-", color="#3b7dd8", label="wobble")
    if np.isfinite(target_rp):
        ax[0].axhline(target_rp, color="#d8743b", ls="--", label="circus-omp [0.8,inf]")
    if T_h is not None:
        ax[0].axvline(T_h["threshold"], color="0.4", ls=":", label=f"T_h={T_h['threshold']:.2g}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("wobble threshold (raw^2)")
    ax[0].set_ylabel("median rp_contamination (>=50-spk units)")
    ax[0].set_title(f"Precision vs threshold @ {h:.0f}h\n(match wobble <= circus = matched precision)")
    ax[0].legend(fontsize=8)
    ax[1].plot(thr_arr, cov_arr, "o-", color="#3b7dd8", label="wobble >=12 MAD cov")
    ax[1].plot(thr_arr, spur_arr, "s--", color="#7d3bd8", label="wobble spurious %")
    ax[1].axhline(m_c["coverage"][">=12_pooled"], color="#d8743b", ls="--", label="circus cov")
    ax[1].axhline(m_c["spurious_frac"] * 100, color="#d87b3b", ls=":", label="circus spurious")
    if T_h is not None:
        ax[1].axvline(T_h["threshold"], color="0.4", ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("wobble threshold (raw^2)")
    ax[1].set_ylabel("%")
    ax[1].set_title("Coverage (>=12 MAD) + spurious vs threshold")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figp = WV / f"threshold_calib_w{int(h)}h.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)

    out = {"window_h": h, "win_s": WIN_S, "n_bank_units": int(bank.unit_ids.size), "tsq_median": med,
           "circus": {"method_kwargs": {"amplitudes": [0.8, None]}, "n_spikes": int(sp_c.size), **m_c},
           "target_median_rp_contamination": target_rp,
           "grid_factors": GRID_FACTORS, "wobble_runs": wobble_runs,
           "chosen_T_h": (T_h["threshold"] if T_h else None),
           "chosen_factor": (T_h["factor"] if T_h else None)}
    (WV / f"threshold_calib_w{int(h)}h.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / f'threshold_calib_w{int(h)}h.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

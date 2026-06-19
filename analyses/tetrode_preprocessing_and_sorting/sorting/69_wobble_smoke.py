"""De-risk step 1 for the wobble-vs-circus head-to-head: does SI's `wobble` matcher run on our
geometry-free, raw-unit, group-sparse tetrode bank, and on what threshold SCALE?

Wobble detects on the normalized objective ``2*conv - ||t||^2`` (units of amplitude^2). Our templates
are RAW units (gain 0.195), so the SI default threshold=50 (Neuropixels uV) is meaningless here. At a
clean full-amplitude match conv ~= ||t||^2, so the NATURAL threshold scale is the per-template ||t||^2
distribution -- which we compute directly from the bank (no need to crack open wobble's internals).
This script:
  * builds the shared MS5 bank for the busiest task-1 window (h=26),
  * prints ||t||^2 quantiles (bounds the calibration grid for script 70),
  * runs wobble at a few thresholds spanning that scale (confirms it runs, no error, spikes>0,
    and that spike count is monotone-decreasing in threshold = the scale is right),
  * runs one threshold TWICE to confirm engine="numpy" determinism,
and writes wobble_vs_circus/smoke_w26h.json. No scoring yet (that is 70/71).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/69_wobble_smoke.py
"""
import json
import pathlib
import shutil
import time

import numpy as np

from _mp_common import build_templates_object, materialize_span, run_matching, wobble_method_kwargs
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_H = 26.0   # busiest task-1 window (sleep-dep) -> most stringent
WIN_S = 1800.0
N_JOBS = 16


def template_norms_sq(templates):
    """||t||^2 per unit (sum of squares over samples x active channels of the packed sparse template).

    Unused channel slots in the packed (n_units, n_samp, max_active) array are zeros, so they add
    nothing to the per-unit sum -- this equals the visible-channel ||t||^2 wobble uses internally.
    """
    arr = np.asarray(templates.templates_array, dtype=np.float64)  # (n_units, n_samp, max_active)
    return (arr ** 2).sum(axis=(1, 2))


def main():
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    a = int(WIN_H * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a, b)
    win.reset_times()

    sdir = WV / f"w{int(WIN_H)}h" / "ref_sort"
    shutil.rmtree(sdir, ignore_errors=True)
    t0 = time.perf_counter()
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=True, n_jobs=N_JOBS, seed=0)
    t_bank = time.perf_counter() - t0
    assert bank.are_templates_sparse(), "bank must be group-sparse for wobble"

    tsq = template_norms_sq(bank)
    qs = {q: float(np.quantile(tsq, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}
    med = qs[0.5]
    print(f"window @ {WIN_H:.0f}h: {bank.unit_ids.size} units, bank built in {t_bank:.0f}s", flush=True)
    print("||t||^2 quantiles (raw^2): " + "  ".join(f"p{int(q*100)}={v:.3g}" for q, v in qs.items()),
          flush=True)

    # thresholds spanning the ||t||^2 scale: a clean match peaks the normalized objective near ||t||^2,
    # so spike count should fall as threshold rises across [~0.05*med .. ~2*med].
    grid = [round(f * med, 3) for f in (0.05, 0.25, 0.5, 1.0, 2.0)]
    runs = []
    for thr in sorted(grid, reverse=True):  # high->low: cheap runs first, a runaway low-thr run can't block signal
        t0 = time.perf_counter()
        mp, spikes = run_matching(win, bank, method="wobble",
                                  method_kwargs=wobble_method_kwargs(bank, threshold=thr), n_jobs=N_JOBS)
        dt = time.perf_counter() - t0
        n = int(spikes.size)
        nuniq = int(mp.get_num_units())
        print(f"  thr={thr:>12.3g}  spikes={n:>8,}  active_units={nuniq:>4}  {dt:.0f}s", flush=True)
        runs.append({"threshold": thr, "n_spikes": n, "n_active_units": nuniq, "seconds": round(dt, 1)})
    runs.sort(key=lambda r: r["threshold"])  # ascending for the monotonicity check + JSON

    # determinism: re-run the middle threshold, compare exact spike-sample set
    thr_d = grid[2]
    _, sp1 = run_matching(win, bank, method="wobble",
                          method_kwargs=wobble_method_kwargs(bank, threshold=thr_d), n_jobs=N_JOBS)
    # already have a run at thr_d above; recompute fresh for a clean second sample
    _, sp2 = run_matching(win, bank, method="wobble",
                          method_kwargs=wobble_method_kwargs(bank, threshold=thr_d), n_jobs=N_JOBS)
    s1 = np.sort(sp1["sample_index"].astype(np.int64))
    s2 = np.sort(sp2["sample_index"].astype(np.int64))
    deterministic = bool(s1.size == s2.size and np.array_equal(s1, s2))
    print(f"determinism @ thr={thr_d:.3g}: n1={s1.size:,} n2={s2.size:,} identical={deterministic}",
          flush=True)

    counts = [r["n_spikes"] for r in runs]
    monotone = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    print(f"spike count monotone-decreasing in threshold: {monotone}", flush=True)

    out = {
        "window_h": WIN_H, "win_s": WIN_S, "n_units": int(bank.unit_ids.size),
        "bank_build_seconds": round(t_bank, 1),
        "tsq_quantiles": qs, "tsq_median": med,
        "threshold_grid": grid, "runs": runs,
        "determinism_threshold": thr_d, "deterministic": deterministic,
        "monotone_decreasing": monotone,
    }
    (WV / "smoke_w26h.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {WV / 'smoke_w26h.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

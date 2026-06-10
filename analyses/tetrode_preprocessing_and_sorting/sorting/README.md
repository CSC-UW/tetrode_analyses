---
title: MountainSort5 spike-sorting study for tetrode session 2026-05-27_09-07-52
scope: tetrode_analyses
status: active
source: code_inspection
created: 2026-06-10
last_updated: 2026-06-10
confidence: high
confirmed_by_user: not_required
---

# Spike sorting (MountainSort5)

Downstream of the compressed Zarr stores (`../compression/`): spike sort the full
48 h session by tetrode group with MountainSort5, then study what perturbs the
sort. Part of the `tetrode_preprocessing_and_sorting` study (see `../README.md`).

## Canonical writeups

- **`SORTING_COMPARISON_FINDINGS.md`** — the main findings doc: lossless-vs-lossy
  agreement, the full-48 h determinism baseline (blosc-vs-blosc = 1.0), the
  perturbation ladder (bps2.25 / int16 / bps6.0), curated agreement on
  well-isolated units, block-duration and training-duration sweeps, the ms5 int32
  ~19.9 h ceiling, the production memory/disk footprint, and the two-RNG
  non-determinism analysis (whitening seed + PCA solver).
- **`ms5_parameter_comparison.md`** — MountainSort5 parameter defaults compared
  across vanilla ms5, the NP-quickstart example, and the SpikeInterface wrapper.
- `mountainsort.pdf` — the MountainSort reference paper.

A closely related upstream investigation (the SpikeInterface `frame_slice` /
`BinaryFolderRecording` memory blow-up, root-caused while measuring
`training_duration` footprint in `23_`) lives in `../si_frame_slice_memory/`.

## Files

| file | purpose |
|---|---|
| `10_sort_ms5.py` / `10b_sort_wavpack.py` | initial MountainSort5 sort by tetrode group of both stores |
| `11_compare_sortings.py` | compare the two stores' sortings (lossless blosc vs lossy wavpack) → `sorting_agreement_matrix.png` |
| `12_sort_seeded_and_compare.py` / `13_sort_wavpack_and_compare.py` | seeded re-sort + clean lossless-vs-lossy comparison (resume after disk abort) |
| `15_determinism_baseline.py` | full-48 h determinism baseline (blosc-A vs blosc-B) + lossless-vs-lossy, post PCA fix |
| `16_sort_int16_and_compare.py` | int16-quantization perturbation test vs float32 reference |
| `18_sort_bps6_and_compare.py` | high-fidelity lossy (bps=6.0) agreement vs lossless |
| `19_metric_distributions.py` | quality-metric distributions on the lossless reference (`metric_distributions_blosc-A.{png,csv}`) |
| `20_curated_agreement.py` | curated lossless-vs-perturbed agreement on well-isolated units |
| `21_sort_blocksize_and_compare.py` / `22_sort_longblocks.py` | scheme-3 block-duration sweep (900 s → 21600 s) + agreement vs 3600 s |
| `23_bench_training_duration.py` | scheme-2 `training_duration_sec` runtime + peak-memory footprint; surfaces the int32 ceiling |
| `24_sort_12h_blocks_1h_train.py` | full-48 h production sort at 12 h blocks / 1 h training; end-to-end footprint |
| `25_build_analyzer_12hblock_train1h.py` | build the Zarr `SortingAnalyzer` for the 12 h-block / 1 h-training sort (13 extensions; tetrode-group sparsity) → `blosc-43200s-train3600s/analyzer.zarr` |
| `probe_pca_solver.py` / `probe_pca_solver_np.py` | instrumented `(L, n_features, solver)` probes (tetrode + Neuropixels) for the PCA-nondeterminism analysis |

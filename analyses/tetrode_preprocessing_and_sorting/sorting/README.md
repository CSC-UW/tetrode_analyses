---
title: MountainSort5 + Kilosort4 spike-sorting study for tetrode session 2026-05-27_09-07-52
updated: 2026-06-11
---

# Spike sorting (MountainSort5)

Downstream of the compressed Zarr stores (`../compression/`): spike sort the full
48 h session by tetrode group with MountainSort5, then study what perturbs the
sort. Part of the `tetrode_preprocessing_and_sorting` study (see `../README.md`).

A second sorter, **Kilosort4** (`30_`–`32_`), was attempted on the same lossless store
with the same preprocessing (bandpass 300–6000 Hz + global CMR), no drift correction,
by tetrode group, for comparison against the MS5 single-block (scheme-2) sort. **It does
not complete at 48 h on the 32 GB GPU** — KS4's final clustering allocates an
`(n_spikes × n_clusters)` dense tensor that needs ~42 GiB for one tetrode's ~24 M
48 h spikes. See **`KS4_FEASIBILITY_FINDINGS.md`** for the full diagnosis, the install
recipe, and what would fit (≤24 h window, higher thresholds, time-block+stitch, bigger
GPU). `refs/kilosort4.pdf` is the reference paper.

## Canonical writeups

- **`SORTING_COMPARISON_FINDINGS.md`** — the main findings doc: lossless-vs-lossy
  agreement, the full-48 h determinism baseline (blosc-vs-blosc = 1.0), the
  perturbation ladder (bps2.25 / int16 / bps6.0), curated agreement on
  well-isolated units, block-duration and training-duration sweeps, the ms5 int32
  ~19.9 h ceiling, the production memory/disk footprint, and the two-RNG
  non-determinism analysis (whitening seed + PCA solver).
- **`ms5_parameter_comparison.md`** — MountainSort5 parameter defaults compared
  across vanilla ms5, the NP-quickstart example, and the SpikeInterface wrapper.
- **`KS4_FEASIBILITY_FINDINGS.md`** — why Kilosort4 cannot complete a 48 h tetrode
  sort on the 32 GB GPU (the clustering memory wall), the install recipe, the
  failure ladder, the measured footprint vs MS5, and what would fit.
- `mountainsort.pdf` / `refs/kilosort4.pdf` — the sorter reference papers.

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
| `27_sort_48h_singleblock_scheme2.py` | full-48 h sort as a single scheme-2 block / 1 h training (large-block limit; counterpart to `24_`) + footprint → `blosc-scheme2-train3600s/` |
| `28_build_analyzer_singleblock_scheme2.py` | build the Zarr `SortingAnalyzer` for the single-block scheme-2 sort (13 extensions; tetrode-group sparsity) → `blosc-scheme2-train3600s/analyzer.zarr` |
| `29_compare_singleblock_vs_12hblock.py` | agreement of single-block scheme 2 vs the 12 h-block scheme-3 sort, raw + curated tiers → `comparison_singleblock_vs_12hblock_summary.json`, `agreement_singleblock_vs_12hblock.png` |
| `30_sort_ks4.py` | full-48 h **Kilosort4** sort by tetrode group, NO drift, same preprocessing as MS5; footprint incl. GPU. **Does NOT complete at 48 h** — OOMs at clustering (see `KS4_FEASIBILITY_FINDINGS.md`); runnable for ≤24 h windows |
| `31_build_analyzer_ks4.py` | (staged, never run) Zarr `SortingAnalyzer` for a KS4 sort (13 extensions; tetrode-group sparsity; counterpart to `28_`) — ready if a completed KS4 sort exists |
| `32_compare_ks4_vs_ms5.py` | (staged, never run) agreement of KS4 vs MS5 (scheme-2 single block), raw + curated tiers — ready if a completed KS4 sort exists |
| `probe_pca_solver.py` / `probe_pca_solver_np.py` | instrumented `(L, n_features, solver)` probes (tetrode + Neuropixels) for the PCA-nondeterminism analysis |

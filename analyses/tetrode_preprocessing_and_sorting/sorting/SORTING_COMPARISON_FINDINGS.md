---
title: MountainSort5 sorting + lossless-vs-lossy comparison (2026-05-27_09-07-52)
updated: 2026-06-03
---

# Sorting both compression stores + comparison

Full 48.29 h session (`2026-05-27_09-07-52`, experiments 1+2 concatenated),
16 tetrodes, sorted with **MountainSort5 scheme 3** from each compression store
(blosc-zstd lossless vs wavpack-bps2.25 lossy), then compared.

Three runs exist. The **post-fix run is canonical** (`sortings_seed42_pcafix/`,
below): it controls BOTH nondeterminism RNGs (`whitening_seed=42` + the
deterministic-PCA solver) and includes the decisive blosc-vs-blosc determinism
control. The seeded-only run (`sortings_seed42/`, deleted 2026-06-03) and the
original unseeded cross-tetrode-CMR run (`sortings/`, deleted 2026-06-01) are
kept below only as labeled contrasts; their on-disk outputs are gone.

## Canonical run: post-fix determinism baseline (`whitening_seed=42` + deterministic PCA)
Script: `15_determinism_baseline.py` (three full-session sorts: blosc twice for
the determinism ceiling, wavpack once for lossless-vs-lossy), `tetrode_analyses.sorting`
(pipeline). Outputs in `sortings_seed42_pcafix/`. Same pipeline as the seeded run
below, but the active MountainSort5 is the editable checkout carrying the
deterministic-PCA fix (Sources 2 + 2b). The deterministic bandpass+global-CMR
materialize was computed once for blosc and reused for both blosc sorts (shared
`cmr_cache_dir`); only the sorter's own RNG differs between blosc-A and blosc-B.

### Determinism ceiling — blosc-A vs blosc-B (same lossless store, sorted twice)
| unit set | blosc-A | blosc-B | matched | mean agreement |
|---|---|---|---|---|
| all units | 781 | 781 | 781 | **1.0** |
| FR ≥ 0.1 Hz | 317 | 317 | 317 | **1.0** |
| FR ≥ 0.5 Hz | 139 | 139 | 139 | **1.0** |
| FR ≥ 1.0 Hz | 96 | 96 | 96 | **1.0** |

Per-tetrode counts are **bit-identical** between A and B. **The full-48 h sort is
now perfectly reproducible** — Sources 1+2+2b fully determinize it, and the
residual Source-3 (FP/threading) noise did not flip a single unit at the match≥0.5
threshold. This is the decisive control that was missing: the 0.996 slice ceiling
is superseded by an exact **1.0 at full scale**. (Sort times 139.3 / 105.6 min;
the second is faster only from a warm cache — identical output.)

### blosc (lossless) vs wavpack (lossy) agreement — now with ceiling = 1.0
| unit set | blosc | wavpack | matched | mean agreement |
|---|---|---|---|---|
| all units | 781 | 671 | 278 | 0.708 |
| FR ≥ 0.1 Hz | 317 | 262 | 139 | 0.685 |
| FR ≥ 0.5 Hz | 139 | 124 | 84 | 0.682 |
| FR ≥ 1.0 Hz | 96 | 88 | 65 | 0.700 |

Numerically ~unchanged from the seeded-only run (~0.706), but now **interpretable**:
since blosc-vs-blosc = 1.0, the full ~0.29 shortfall is no longer confounded by
sorter variability. (wavpack sort 188.1 min.) See caveats below for the framing
(user-confirmed 2026-06-03: the gap is a clean compression effect on the *raw*
sort; whether it touches well-isolated units is the open question, pending curation).

### int16 quantization of the preprocessed binary vs float32 (`16_sort_int16_and_compare.py`)
Same blosc store, same seed + deterministic PCA, but the materialized bandpass+CMR
binary is written as **int16** instead of float32 — i.e. the sorter input is
quantized to integer ADC-count resolution (the original acquisition resolution;
post-CMR std ≈ 80 counts, no clipping, ~48 dB quant SNR; SI casts via `astype`,
truncation toward zero, so a slight ≤1-count bias toward zero). Compared against
the float32 `blosc-A` reference:

| unit set | float32 | int16 | matched | mean agreement |
|---|---|---|---|---|
| all units | 781 | 783 | 406 | 0.780 |
| FR ≥ 0.1 Hz | 317 | 302 | 193 | 0.783 |
| FR ≥ 0.5 Hz | 139 | 137 | 102 | 0.799 |
| FR ≥ 1.0 Hz | 96 | 92 | 75 | 0.815 |

This required fixing a real SpikeInterface bug first: `write_binary` sized the file
and per-chunk seek offsets by the *source* dtype while casting to the *target*
dtype, so `save(dtype="int16")` produced a corrupt 2×-too-large binary (the
materialize aborted at a `set_times` assertion). Fix + regression test in the SI
checkout (`core/time_series_tools.py`, `test_write_binary_dtype_conversion`).

### Perturbation ladder (all vs the deterministic float32 sort, match≥0.5, all units)
| perturbation | in-band err vs noise | agreement | matched | Δ from ceiling | agreement FR≥1 Hz |
|---|---|---|---|---|---|
| none (blosc-vs-blosc, determinism ceiling) | 0 | **1.000** | 781/781 | — | 1.000 |
| wavpack lossy (bps=6.0) | 0.01× (0.18 µV) | **0.780** | 423 | 0.220 | 0.830 |
| int16 quantization (integer ADC counts) | ~1-count trunc | **0.780** | 406 | 0.220 | 0.815 |
| wavpack lossy (bps=2.25) | 0.19× (3.41 µV) | **0.708** | 278 | 0.292 | 0.830 |

**Key finding — agreement plateaus at ~0.78 and does NOT climb toward 1.0 as
fidelity increases.** bps=6.0's in-band error is ~19× smaller than bps=2.25's
(0.01× vs 0.19× the noise floor — near-lossless), yet its agreement (0.780) is
identical to int16's and only marginally above bps=2.25's 0.708. So the ~0.22
shortfall from the determinism ceiling is **not driven by the magnitude of the
compression error** — even a sub-noise (0.01×) perturbation flips roughly the
same fraction of units as a 0.19× one. This is the signature of a **raw,
over-split sort dominated by marginal fragment/noise units sitting on clustering /
cross-block-label-matching decision boundaries**: any input perturbation, however
tiny, reshuffles them. (Consistent with the determinism finding that the sort is
bit-exact only when the input is bit-identical.)

**Implication (NOT user-confirmed science):** raising `bps` does **not** recover
agreement — bps=6.0 buys near-zero error but only 2.66× compression (~233 GB vs
blosc's ~325 GB) and still lands at 0.78. So the lossy-vs-lossless gap is an
artifact of the uncurated sort's instability, not of compression fidelity per se.
The decision-relevant question is therefore **curation** (resolved below): the raw
plateau is a fragment artifact; on well-isolated units the fidelity ordering
re-emerges.

## Curated agreement — well-isolated units (`19_`, `20_`)
Quality metrics computed on the lossless reference `blosc-A` (781 units,
`19_metric_distributions.py`; `metric_distributions_blosc-A.{png,csv}`) showed that
the lab's default tiers don't transfer to this 48 h scheme-3 sort: **`snr`**
(median 9.7) and **`amplitude_cutoff`** (≈0) don't discriminate (all units are loud
and complete), and **`presence_ratio`** (median **0.04**) collapses any cutoff —
but that reflects scheme-3's per-block units not being merged across the 48 h, i.e.
tracking duration, **not** isolation. The metrics that actually separate are
refractory: **`isi_violations_ratio`** (median 1.29 — the median raw unit is
contaminated) and **`rp_contamination`**.

"Well-isolated" was therefore defined (user-confirmed 2026-06-05) on
`isi_violations_ratio` + `rp_contamination` + a `firing_rate` floor, three tiers,
`presence_ratio` dropped. Good-unit counts on `blosc-A`: **permissive 104 /
moderate 61 / conservative 30** (of 781). Each perturbed sort was matched against
`blosc-A`, then restricted to the good reference units (`20_curated_agreement.py`).

*mean agreement among matched / fraction of good units reproduced (match≥0.5):*
| reference set | n | bps2.25 | int16 | bps6.0 |
|---|---|---|---|---|
| all units (raw) | 781 | 0.71 / 0.36 | 0.78 / 0.52 | 0.78 / 0.54 |
| permissive | 104 | 0.71 / 0.72 | 0.82 / 0.81 | 0.83 / 0.86 |
| moderate | 61 | 0.72 / 0.84 | 0.85 / 0.82 | 0.87 / 0.85 |
| conservative | 30 | 0.77 / 0.90 | 0.86 / 0.87 | 0.88 / 0.90 |

**Interpretation (framing user-confirmed 2026-06-05):**
1. **Well-isolated units are largely preserved.** Reproduction (match fraction) rises
   from 0.36–0.54 (raw, fragment-dominated) to **0.85–0.90** at moderate/conservative —
   the low raw numbers were over-split fragments, not loss of real units.
2. **Fidelity matters on good units** — the raw "plateau" was an artifact. Among matched
   good units, bps=2.25 (**0.72–0.77**) is clearly worse than int16 (~0.85) and bps=6.0
   (**0.87–0.88**); more bits → faithfully preserved spike trains.
3. **But agreement caps ~0.88 even near-lossless** (bps6.0, 0.01× noise). Even
   well-isolated units' spike trains are perturbed ~12% by *any* lossy compression
   (the determinism ceiling is 1.0, so this is purely compression). So lossy is not
   "free" even for good units — there is an irreducible ~0.12 perturbation.
4. **Decision support:** the original "bps=2.25 too low" judgment holds even on good
   units (0.72–0.77). Raising to bps=6.0 lifts good units to ~0.88 but only buys 2.66×
   (~233 GB) vs lossless 2.05× (~325 GB) — a modest ratio gain for a still-imperfect
   match. Whether ~0.88 clears the production bar is the user's call; if not, lossless
   is the safer choice for roughly the same size.

## Block-duration sensitivity (`21_sort_blocksize_and_compare.py`, `22_sort_longblocks.py`)
Lossless blosc re-sorted at `scheme3_block_duration_sec` from 900 s to 21600 s,
everything else fixed (global CMR, seed 42, deterministic PCA). Shared float32
materialize (block duration only affects ms5's sorting stage). Determinism ceiling
is 1.0, so any disagreement vs the 3600 s reference (`blosc-A`) is purely the
block-duration parameter.

**Unit counts drop monotonically as blocks lengthen** (fewer blocks → fewer
unmerged per-block fragments); **pure sort time is roughly flat** (~100–125 min —
block duration mostly redistributes the same total detect/classify work):

| block | blocks over 48 h | units | pure sort time | training window |
|---|---|---|---|---|
| 900 s | 193 | 2462 | ~125 min | 300 s (default) |
| 1800 s | 97 | 1363 | ~112 min | 300 s (default) |
| 3600 s | 48 | 781 | ~106 min | 300 s (default) |
| 7200 s | 24 | 457 | 115.6 min | 300 s (default) |
| 14400 s | 12 | 307 | 102.8 min | 300 s (default) |
| 21600 s | 8 | 238 | 106.5 min | 300 s (default) |
| 43200 s (12 h) | 4 | 232 | 93.6 min | **3600 s (1 h)** |

238 units at 6 h ≈ ~15/tetrode — finally in the plausible isolatable-unit range for
16 tetrodes, i.e. long blocks sharply reduce the over-splitting. (`22_` reports
time/units only; `21_` adds the agreement-vs-3600 s comparison below. Long-block
sortings: `blosc-{7200,14400,21600}s/`; `sorting_longblocks_summary.json`.)

**The yield-vs-block curve flattens at multi-hour blocks** (`24_sort_12h_blocks_1h_train.py`):
doubling 6 h → 12 h barely moved the count (238 → 232), versus the steep drops at
short blocks (900 → 1800 → 3600 s ≈ halving each step). By multi-hour blocks the
per-block sort already captures the dominant well-isolated units; the marginal
candidates are merged across the long stationarity window. Caveat: the 12 h run used
a **1 h training window** while the rest of the series used the 300 s default, so
232 vs 238 is not a controlled training comparison — but it confirms 12 h blocks land
in the same ~230-unit regime, and widening training 300 s → 1 h did **not** materially
change yield (consistent with training duration being second-order to block duration;
see the footprint note below). Sorting: `blosc-43200s-train3600s/`;
`sorting_12hblock_train1h_summary.json`.

*agreement vs 3600 s (mean agreement among matched / fraction reproduced):*
| reference set | n | 1800 s | 900 s |
|---|---|---|---|
| all units (raw) | 781 | 0.70 / 0.44 | 0.69 / 0.36 |
| permissive | 104 | 0.76 / 0.79 | 0.76 / 0.73 |
| moderate | 61 | 0.78 / 0.87 | 0.77 / 0.84 |
| conservative | 30 | 0.78 / 0.87 | 0.80 / 0.80 |

**Interpretation (NOT user-confirmed science):** the block duration is a
**first-order knob**, and changing it perturbs the sort *more than near-lossless
compression does*. Placing the curated-moderate mean-agreement numbers on one scale
(same reference, same metric):

| change vs 3600 s lossless | curated mean agreement (matched) |
|---|---|
| none (re-run, determinism ceiling) | 1.00 |
| bps=6.0 (near-lossless compression) | 0.87 |
| int16 quantization | 0.85 |
| **block 1800 s / 900 s** | **0.78 / 0.77** |
| bps=2.25 (aggressive lossy) | 0.72 |

So halving (or quartering) the block duration moves the good-unit sort by *more*
than near-lossless compression and about as much as aggressive bps=2.25 lossy
compression. Most good units are still found (match fraction 0.73–0.87), but their
spike trains differ ~0.22. 1800 s and 900 s are similar to each other (~0.77) — the
shift happens on leaving 3600 s, not in the further halving (though unit count keeps
climbing). Takeaway: an internal sorter parameter introduces variability comparable
to the compression choice under study — reinforcing that the raw scheme-3 sort is
sensitive to many choices, and that production stability requires fixing parameters
*and* curating.

## training_duration footprint & the ms5 int32 ceiling (`23_bench_training_duration.py`)
Measured runtime + peak memory of scheme-2 `training_duration_sec` on tononi-2
(1.5 TiB RAM, 112c/224t), sorting a genuine T-second crop of all 16 tetrodes at
production `n_jobs=5`. Memory = **peak USS** (private working set summed over the
process tree; USS is the right metric because RSS additionally counts the mmap'd
binary's reclaimable file cache). 4-ch tetrodes (denser probes scale up).

Methodology note (root-caused via staged synthetic reproductions, `diag_framesize*.py`,
now in the sibling `../si_frame_slice_memory/`; full writeup in
`../si_frame_slice_memory/frame_slice_memory_FINDINGS.md`):
an earlier attempt that frame-sliced the *full 48 h* materialized binary (rather than
materializing a genuine T-second crop) showed ~500 GB for a nominal 300 s of sorting.
Cause: in the **`run_sorter_by_property` + `BinaryFolderRecording` (the class
`recording.save(format="binary")`/`si.load` produces) + `frame_slice`** path, each
worker's memory scales with the **full parent recording**, not the frame-sliced
length — ~full-per-tetrode (4 ch × 48 h × f32 ≈ 83 GB) × `n_jobs`(5) ≈ 415 GB, plus a
secondary ~42 GB from the full-length float64 time vector that `frame_slice` carries
at parent length. `frame_slice`'s *length* is honored (the sort correctly processes
300 s and yields ~300 s of units) but its *memory* is full-parent-scaled. Verified:
the single-file `BinaryRecordingExtractor` path does **not** exhibit this (~1.8 GB
even at the exact 48 h frame count, >2³²); the file-backed `BinaryFolderRecording`
path does (~full-data × workers, time-vector-independent — persists under
`reset_times`). So it's an apparent **SpikeInterface inefficiency** (worker memory not
bounded by the frame slice on this path), worth a minimal-repro issue; the exact
internal trigger was not isolated to a line (`is_binary_compatible` is identical for
both classes, so that's not it). Materializing a genuine per-T crop binary avoids it
(full == crop) and gives the clean, monotonic numbers below — the reliable measurement.

| training_duration | sort time | peak USS | USS/tetrode | units |
|---|---|---|---|---|
| 300 s | 0.2 min | 1.8 GB | 0.36 GB | 86 |
| 600 s | 0.3 min | 3.0 GB | 0.59 GB | 99 |
| 1200 s | 0.7 min | 4.9 GB | 0.98 GB | 107 |
| 2400 s | 1.2 min | 8.5 GB | 1.71 GB | 110 |
| 4800 s (1.3 h) | 2.7 min | 17.0 GB | 3.39 GB | 113 |
| 9600 s (2.7 h) | 4.9 min | 29.8 GB | 5.96 GB | 109 |
| 19200 s (5.3 h) | 10.4 min | 62.8 GB | 12.57 GB | 109 |
| 38400 s (10.7 h) | 21.3 min | 122.1 GB | 24.43 GB | 115 |
| 76800 s (21.3 h) | — | — | — | **CRASH** |

Peak USS and sort time both scale **~linearly** with training_duration; unit count
plateaus ~110 (the 16 tetrodes' isolatable-unit count, found from any ≥~1 h window).

**The ceiling is a software limit, not the hardware.** At 76800 s the run crashed with
`OverflowError: ... out of bounds for int32` in `mountainsort5/core/detect_spikes.py:78`
(`times = np.array(times, dtype=np.int32)`): ms5 indexes spike sample-times as **int32**,
which overflows past **2³¹ = 71,583 s ≈ 19.9 h**. So the largest workable
`training_duration` (or scheme-3 `block_duration`) is **~19.9 h**, where the
*extrapolated* footprint is only ~230 GB — far below the 1.5 TiB RAM and the 500 GB
courteous budget. Memory never binds first: even the full 48 h as one training window
would be ~550 GB USS (fits 1.5 TiB) **if the int32 were widened to int64**. Recommended
fix: int64 sample indices in `detect_spikes` (and audit other int32 sample-index sites)
to lift the cap to the full recording. Normal runs (blocks ≤ 6 h) never hit it.

### Full-48 h production footprint at 12 h blocks / 1 h training (`24_sort_12h_blocks_1h_train.py`)
The crop sweep above isolates the *training* cost; this is the real end-to-end
footprint of an actual 48 h scheme-3 sort. Full 48 h, `block_duration=43200 s` (12 h,
4 blocks), `training_duration=3600 s` (1 h, uniform), global CMR, seed 42, `n_jobs=5`.
Both 12 h blocks and the 1 h window are under the 19.9 h int32 ceiling, and phase-2
classification is chunked (~833 s/chunk at 4 ch) so no single `detect_spikes` call
approaches 2³¹.

| phase | wall time | peak USS | peak RSS (incl. file cache) | peak `/nvme` written |
|---|---|---|---|---|
| materialize (bandpass + global CMR → f32, `n_jobs=96`) | 32.9 min | 83.6 GB | 83.1 GB | 1377 GB |
| sort (16 tetrodes, `n_jobs=5`) | 93.6 min | **90.6 GB** | 267.1 GB | +626 GB |
| **total** | **~126 min (2.1 h)** | ~90 GB | — | ~2.0 TB peak |

- **RAM is not the constraint:** peak private working set ~90 GB (~18 GB/tetrode × 5
  workers) — ~6 % of the 1.5 TiB host. RSS 267 GB is mostly reclaimable mmap'd pages
  of the materialized binary (file cache), not anonymous pressure.
- **Disk is the cost:** the 64-ch f32 materialized cache is ~1.38 TB; the sort adds
  ~626 GB of per-tetrode 4-ch caches ms5 builds on the fly → ~2.0 TB peak on `/nvme`.
- **CPU not reported — sampler metric is broken.** The `TreeSampler` logged 1.4 cores,
  implausible for a 5-way sort: it re-creates child `Process` objects every loop, so
  `cpu_percent()` returns 0 on each first call and only the parent accumulates. Memory/
  disk/time are point-in-time snapshots and unaffected. By design the sort is ~5 busy
  cores (`n_jobs=5`; per-tetrode ms5 work is largely single-threaded). The same bug is
  present in `23_`'s `peak_cpu_cores` column (memory/time there are valid). Fix before
  citing any CPU number: keep a persistent `{pid: Process}` map across loop iterations.

## Prior run (contrast): seeded, global CMR (`whitening_seed=42`, no PCA fix)
Scripts: `12_sort_seeded_and_compare.py` (both stores; produced blosc),
`13_sort_wavpack_and_compare.py` (wavpack-only resume after a shared-disk abort,
reusing the saved blosc sorting), `tetrode_analyses.sorting` (pipeline).
Outputs in `sortings_seed42/`.

Pipeline (per store): bandpass 300–6000 Hz (float32) → **global common median
reference** (`common_reference(reference="global", operator="median")`; chosen
over cross-tetrode CMR after a waveform benchmark showed ~0.25% peak difference
at ~20× lower materialize cost) → **materialize once** to a local float32 binary
→ `split_by("group")` → MountainSort5 per tetrode (`scheme="3"`,
`scheme3_block_duration_sec=3600`, ms5 does per-group whitening, seeded),
parallel via `run_sorter_by_property`.

### Unit counts
| store | sort time | total units | per-tetrode range |
|---|---|---|---|
| blosc-zstd (lossless) | 139.5 min | **785** | 18–108 |
| wavpack-bps2.25 (lossy) | 175.9 min | **677** | 14–79 |

### blosc (lossless) vs wavpack (lossy) agreement (`compare_two_sorters`, match≥0.5)
| unit set | blosc | wavpack | matched | mean agreement |
|---|---|---|---|---|
| all units | 785 | 677 | 277 | 0.706 |
| FR ≥ 0.1 Hz | 320 | 264 | 141 | 0.680 |
| FR ≥ 0.5 Hz | 140 | 123 | 85 | 0.678 |
| FR ≥ 1.0 Hz | 96 | 87 | 66 | 0.694 |

### Contrast: original unseeded run (cross-tetrode CMR; outputs deleted)
all units 1391 / 1181, 393 matched, agreement **0.72**; FR≥0.1: 351/328, 139,
0.69; FR≥1.0: 90/84, 47, 0.72. (Sort times 256 / 380 min, the latter at reduced
parallelism.)

## Interpretation & caveats (NOT user-confirmed science)
- **The decisive control is now run, and the sorter is fully deterministic at
  48 h**: blosc-vs-blosc = **1.0** (exact, bit-identical per-tetrode counts) once
  both whitening (Source 1) and PCA (Sources 2/2b) RNGs are controlled. This
  supersedes the 0.996 600 s slice ceiling — that slice number was an artifact of
  measuring on a single block; the unseeded randomized PCA had been compounding
  across ~48 blocks, which is why it did not transfer to full scale. With the fix,
  there is no full-scale variability left to transfer.
- **Consequence for the cross-store number** (framing user-confirmed 2026-06-03): with
  the determinism ceiling at exactly 1.0, the blosc-vs-wavpack agreement (~0.71,
  278/781 units matched) is **no longer confounded by sorter variability** — the
  entire ~0.29 shortfall is attributable to the **bps=2.25 lossy compression**.
  Taken at face value this is a *substantial* raw-sorting effect (fewer units,
  ~71% match, ~0.71 agreement among matched), which would be in tension with eLife
  110170's report that bps=2.25 leaves sorting metrics ~unchanged. **BUT** this is
  a RAW, uncurated comparison dominated by over-split fragment/noise units (next
  caveat); whether the effect survives curation — i.e. whether it touches
  well-isolated units — is the open scientific question and is **not yet
  established**. The curated comparison must be run before concluding anything
  about compression's scientific impact.
- **Heavy over-splitting**: tens of units/tetrode (raw) is far above the few–~15
  isolatable units expected per tetrode; scheme 3 over a 48 h drifting recording
  yields many fragment/noise units not merged across blocks. The sortings are
  **uncurated**; a curated comparison (ISI/SNR/amplitude via `SortingAnalyzer`,
  keep well-isolated units) would be more meaningful than raw counts.

## Non-determinism analysis (verified) — TWO RNG sources, not one

**Source 1 — whitening (found first, now seeded).** SI `whiten()` estimates the
matrix from *randomly* selected data chunks (`get_random_data_chunks(seed=None)`),
and ms5's wrapper called `whiten(...)` unseeded. Fix (PR-able): added a
`whitening_seed` param to SpikeInterface's MountainSort5 wrapper
(`sorters/external/mountainsort5.py`; default `None` = unchanged), passed into the
existing `whiten(...)` call. On a 600 s slice, two seeded runs gave **97/97 units
matched, agreement 0.9959**.

**Source 2 — sklearn PCA randomized SVD (found later, was the missing piece).**
`SnippetClassifier.fit()` calls `decomposition.PCA(n_components=12)` with no
`svd_solver` → sklearn's `'auto'` picks the **stochastic `randomized`** solver
(`random_state=None`, global RNG) whenever `500 < n_samples < 10·n_features`. With
`n_features = T·M = 40·4 = 160` and per-classifier `n_samples` (L) typically
200–1600, this fires constantly. **Measured** (instrumented `fit()`, blosc, 64
fits/run): on the 600 s slice **61% of classifier fits used `randomized`** (39/64;
L median 600, max 1327); the 3600 s slice was nearly identical (56%, L governed by
the fixed 300 s training window, not block length). So Source 2 was live even in
the determinism slice — the 0.996 ceiling held *despite* it because a single
block's perturbation is tiny; across the full run's ~48 blocks (~39 unseeded
`randomized` fits each, feeding the threshold-gated cross-block label matcher) it
compounds. This is why the 0.996 slice ceiling does **not** transfer to 48 h.
Fix (implemented, PR-prep): pick the solver explicitly via a shared helper
`mountainsort5/core/pca_solver.py:deterministic_pca_solver(n_samples, n_features)` —
`covariance_eigh` when `n_samples >= n_features` (≤ an 8000-feature cap), else
exact `full`. Both are exact + deterministic, so the choice is purely speed; the
crossover is measured at `L ≈ n_features` (covariance_eigh time is flat in L,
dominated by the O(n_features³) eigh; full grows linearly in L). For
`SnippetClassifier` `n_features` ≤ ~1400, so it's sub-second and always exact.
Validated bit-reproducible + exact vs `full` (subspace err ~1e-7); regression test
`test_snippet_classifier_is_deterministic`.

**Not tetrode-specific — confirmed on Neuropixels.** Probed the same `fit()` on
CNPIX12-Santiago imec0 (384 ch) using the ms5 NP-quickstart params
(`snippet_mask_radius=60` → neighborhood M=3–5, `n_features`=165/220/275,
`classifier_npca=10`, `max_num_snippets_per_training_batch=1000`): 383 fits,
L median 2049 (max 9332). `'auto'` would pick **`randomized` for 65%** of fits,
`covariance_eigh` for 35% (the L≥10·n_features ones) — essentially the tetrode's
61%. NP's larger L (~3×) is offset by larger `n_features` (raising the
`covariance_eigh` threshold), so the randomized fraction barely moves. The fix
forced `covariance_eigh` for all 383 NP fits (validated end-to-end on real data).
Probe data: `pca_solver_probe_np_quickstart.csv`.

**Source 2b — phase-1 PCA in `compute_pca_features` (same bug, different site,
distinct fix).** `mountainsort5/core/compute_pca_features.py` also calls
`decomposition.PCA(n_components=...)` with no `svd_solver`/`random_state`. It runs
in scheme-1 phase-1 (and the isosplit subdivision) with `npca = npca_per_channel ·
M_all` (e.g. 3·383 = **1149**) on a **full-probe** snippet matrix
(`L × (T·M_all)`, e.g. ~272k × 21065). There `n_features` is huge (~21065), so
`'auto'` also picks **unseeded `randomized`** — a second nondeterminism source.
Fix (implemented, PR-prep): the **same shared helper** `deterministic_pca_solver`
handles it — but because `n_features` (~21065) exceeds the 8000 cap (exact solvers
intractable: ~3.6 GB covariance / `O(n_features³)` eigh; `full` is `O(L·n_features²)`,
worse), it returns **seeded `randomized`** (`random_state=0`). That keeps the exact
same algorithm/cost ms5 already runs and only makes it **reproducible** (up to
Source 3 FP noise — it stays approximate, which is unavoidable at this scale). So
one helper covers both sites: tetrode phase-1 (D=160) → exact `covariance_eigh`,
NP whole-probe phase-1 (D≈21k) → seeded `randomized`. Regression tests:
`test_deterministic_pca_solver_selection`, `test_compute_pca_features_is_deterministic`.
This same call is also (separately) the **OpenBLAS-after-`fork()` deadlock** culprit:
run right after SI's fork-based `recording.save`, its large threaded SVD hangs
(0% CPU, `futex_wait`). Orthogonal workaround that keeps full BLAS threading:
`with threadpoolctl.threadpool_limits(1): recording.save(...)` — see
`gfys_workspace/docs/developer_notes/blas_fork_deadlock.md`.

**Source 3 — floating-point / threading (unseedable, but empirically negligible
at the unit level).** Multithreaded BLAS/LAPACK (PCA SVD, covariance) and
reductions accumulate in thread-schedule-dependent order → ~1e-6 wobble;
`np.argsort` (unstable) on tied spike times could then resolve differently.
`OMP_NUM_THREADS=1` would make it bit-exact at a large speed cost — **but it
turned out not to be needed**: the full-48 h blosc-vs-blosc run reached exactly
1.0 (781/781 units, bit-identical per-tetrode counts) with normal multithreaded
BLAS, so Source 3 did not perturb any final unit at the match≥0.5 level. It
remains the only theoretical residual after Sources 1–2/2b are fixed.

Otherwise verified deterministic: isosplit6 (no RNG, by grep), the scheme-2
training subsample (`'initial'`/`np.linspace`), bandpass, the median CMR, the
materialize. Solver-probe data: `pca_solver_probe_600s.csv` (rows 1–64 = 600 s slice, 65–128 =
3600 s slice; cols: L, T, M, n_features, n_components, n_training_batches, svd_solver).

## Recommended next steps
1. ~~Apply the Source-2 fix, then run the determinism baseline (decisive).~~
   **DONE (2026-06-03)**: full-48 h blosc-vs-blosc = **1.0** with the deterministic-PCA
   fix + `whitening_seed=42`. The sorter is fully reproducible at scale; Source 3
   never manifested. The cross-store gap is now a clean compression measurement.
2. **Curate (ISI violations, SNR, amplitude) and re-compare on well-isolated units**
   — now the priority. The raw ~0.71 cross-store number is dominated by over-split
   fragment/noise units; the open scientific question is whether the bps=2.25 effect
   touches good units. (`SortingAnalyzer` on `sortings_seed42_pcafix/`.)
3. Consider tuning scheme 3 to reduce over-splitting on 48 h data.

## Outputs (`/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/`)
- `*.wavpack-bps2.25.zarr` (94 GB), `*.wavpack-bps6.0.zarr` (233 GB), `*.blosc-zstd.zarr` (377 GB), `*.lfp.zarr` (625 Hz)
- `sortings_seed42_pcafix/{blosc-A,blosc-B,wavpack-bps2.25,blosc-int16,wavpack-bps6.0}/aggregated/` + `by_group/`
  (canonical post-fix run; `blosc-int16` = int16-materialize quantization test;
  `wavpack-bps6.0` = high-fidelity lossy test, `18_sort_bps6_and_compare.py`)
- `sortings_seed42_pcafix/sorting_summary.json`, `sortings_seed42_pcafix/comparison_summary.json`,
  `sortings_seed42_pcafix/comparison_int16_summary.json`, `sortings_seed42_pcafix/comparison_bps6_summary.json`,
  `sortings_seed42_pcafix/comparison_curated_summary.json` (curated, well-isolated)
- curation: `metric_distributions_blosc-A.{png,csv}`, `metric_distributions_summary.json`
  (`19_metric_distributions.py`); curated agreement `20_curated_agreement.py`
- block-duration: `sortings_seed42_pcafix/{blosc-1800s,blosc-900s}/` + `comparison_blocksize_summary.json`,
  `sorting_blocksize_summary.json` (`21_sort_blocksize_and_compare.py`);
  long blocks `sortings_seed42_pcafix/blosc-{7200,14400,21600}s/` + `sorting_longblocks_summary.json`
  (`22_sort_longblocks.py`)
- 12 h block / 1 h training (full-48 h production footprint):
  `sortings_seed42_pcafix/blosc-43200s-train3600s/` + `sorting_12hblock_train1h_summary.json`
  (`24_sort_12h_blocks_1h_train.py`)
- training_duration footprint sweep: `sortings_seed42_pcafix/training_duration_bench.json`
  (`23_bench_training_duration.py`)
- `slice_table.csv`, `conversion_results.json`
- (`sortings_seed42/` — seeded-only run — deleted 2026-06-03; `sortings/` — original
  unseeded run — deleted 2026-06-01)

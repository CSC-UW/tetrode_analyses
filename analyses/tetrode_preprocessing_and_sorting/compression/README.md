---
title: Chunking & compression scheme for tetrode spike sorting
scope: tetrode_analyses
status: active
source: measurement
created: 2026-05-30
last_updated: 2026-06-10
confidence: high
confirmed_by_user: true
---

# Chunking & compression for tetrode recordings → Zarr

Goal: convert ~667 GB of Open Ephys flat-binary continuous tetrode recordings
(e.g. `/Volumes/neuropixel_archive/tetrode_data/2026-05-27_09-07-52`) to a
chunked + compressed format that is efficient to **spike sort by tetrode
group**. Every choice below is backed by a benchmark on a 180 s real sample of
this recording (3×60 s drawn from 10/50/85 % through the file).

## TL;DR recommendation

| Parameter | Value | Why |
|---|---|---|
| Format | **Zarr** (SpikeInterface `ZarrRecordingExtractor`) | native chunking + per-chunk compression; sort-ready, no conversion at read time |
| **Channel chunk** | **4 — one tetrode per chunk** | *the* decisive parameter; see below |
| Probe / channel order | **`attach_tetrode_probegroup()`** | reorders channels to tetrode-contiguous map order (so `channel_chunk_size=4` aligns — this rig's map is **not** identity, TT1 = `.dat` cols 39,37,35,33), sets `group`/`tetrode` properties, attaches a generic-tetrode `ProbeGroup` → sort-ready |
| Time chunk | **30 s** (900 000 samples) | best aligned-read speed, **~54k files/exp**; chunk ≈ 0.5 MB/tetrode compressed |
| Compressor (traces) | **`WavPack(bps=2.25)`** (lossy, default) *or* **`blosc-zstd-l5-bitshuffle`** (lossless) | 7.1× (error ~5× below the spike-band noise floor; matches lab NPX practice) vs 1.73× bit-exact; user-selectable via `--compressor` to compare sorting outcomes |
| Timestamps | **real OE sync clock**, `load_sync_timestamps=True` → Delta-compressed lossless `times_seg0` | SI's default `t_start`+constant-rate is wrong for this rig; the per-sample sync clock is needed for cross-stream alignment |

Run `06_convert.py` (see its docstring) once per compressor to get both stores.
`--bps 0` gives bit-exact lossless WavPack instead.

## Recording facts

- 64 ch, int16 LE, 30 kHz, interleaved sample-major; `bit_volts` = 0.195 µV.
- `experiment1` = 387 GB (28.0 h), `experiment2` = 280 GB → ~667 GB total.
- `sample_numbers` contiguous (no dropped samples), but `timestamps.npy` is a
  **real per-sample hardware sync clock** (sub-sample jitter around 1/fs, first
  sample at ~18.7 s) — it must be preserved for alignment to other streams, not
  discarded as "regular." `OpenEphysBinaryRecordingExtractor(...,
  load_sync_timestamps=True)` loads it via `set_times()`; the zarr writer then
  stores it as a `times_seg0` dataset (Delta-compressed, lossless). Verified:
  exact round-trip, Delta shrinks it **~261×** (e.g. 7.2 MB → 27.6 kB for 30 s).
- **Channel Map is not identity.** Tetrodes are scattered across acquisition
  order, so channels are reordered to `[TT1×4, TT2×4, …]` before chunking —
  otherwise `channel_chunk_size=4` would group unrelated channels. This (plus
  `group`/`tetrode` properties and a tetrode `ProbeGroup`) is done by
  `tetrode_analyses.io.attach_tetrode_probegroup()`, the same helper used in
  `5-19.ipynb`.
- Source read throughput from the archive ≈ 75–109 MB/s ⇒ full conversion is
  **read-bound** (~2 h just to read 667 GB); any codec faster than that per
  core keeps up.

## Evidence

### 1. Channel chunk = 4 is the decisive choice (per-group sorting)

Reading one tetrode (4 ch) from a 64-ch chunk forces decompressing all 64
channels — a **16× read amplification**. Production-threaded, chunk-aligned
group reads (`03_bench_threaded_aligned.py`):

| compressor | shape | aligned group-read | ratio |
|---|---|---|---|
| blosc-zstd-l5-bitshuffle | t30s, **c4** | **1821 MB/s** | 1.73 |
| blosc-zstd-l5-bitshuffle | t30s, c64 | 62 MB/s | 1.70 |
| wavpack | t30s, **c4** | 137 MB/s | 2.05 |
| wavpack | t30s, c64 | 8 MB/s | 2.05 |

`c4` costs **nothing** in compression ratio and gives 1–2 orders of magnitude
faster per-tetrode reads. It also lets the 16 tetrodes sort in parallel over
disjoint chunks.

#### Controlling for chunk size + testing `c1` (`07_bench_chunk_shapes.py`)

The table above varied channel count at equal *time length*, so a c64 chunk
held 16× more bytes than a c4 chunk. To rule out chunk-size confounding, this
benchmark also compares c1/c4/c64 at **equal total chunk bytes** (2.40 MB) and
adds `channel_chunk=1`:

| chunk | blosc read MB/s | blosc ratio | wavpack read MB/s | wavpack ratio | files/exp1 |
|---|---|---|---|---|---|
| c1 (size-matched) | 637 | 1.734 | 47 | 7.064 | 179k |
| **c4 (size-matched)** | **1062** | **1.733** | **93** | **7.090** | 161k |
| c64 (size-matched) | 61 | 1.697 | 12 | 7.039 | 161k |
| **c4 @ t30s** | **1511** | 1.733 | 100 | 7.096 | **54k** |
| c64 @ t30s | 62 | 1.699 | 8 | 7.097 | 3.4k |

Conclusions: (a) c64's slow per-tetrode read (~17× slower than c4) **persists at
equal chunk bytes** — it is genuine channel-read amplification, not a chunk-size
artifact; (b) c64 even compresses slightly *worse* (bitshuffle/decorrelation
spans 64 unrelated channels); (c) **c1 is worse than c4** — ~2× slower reads (4
separate chunk reads per tetrode), more files, and marginally worse WavPack
ratio because c1 forfeits WavPack's inter-channel decorrelation across a
tetrode's 4 contacts. `c4 @ t30s` is the sweet spot: fastest aligned read,
fewest files among c4 options (~54k/exp1).

### 2. Compressor — WavPack wins on size; blosc-zstd on decode speed

36-combo matrix (`02_bench_matrix.py`, `results_round1_matrix.json`), ratios on
this **raw wideband** data (lower than AP-band figures — wideband includes LFP):

| compressor | ratio | note |
|---|---|---|
| **WavPack (lossless)** | **2.05** | best lossless; purpose-built for ephys; 8-thread decode ≈130 MB/s |
| blosc-zstd l9 bitshuffle | 1.75 | +1 % over l5 for ~18× slower write — not worth it |
| blosc-zstd l5 bitshuffle | 1.73 | SI default; decode >1 GB/s |
| zstd l5 + shuffle | 1.72 | |
| blosc-lz4 l5 bitshuffle | 1.61 | fastest, worst ratio |

blosc-zstd-l5-bitshuffle is the right pick if you re-read raw traces often and
want maximum decode speed at the SI default.

**Two selectable stores.** `06_convert.py --compressor {wavpack,blosc-zstd}`
emits either codec on identical chunking/probe/timestamp handling. The intended
workflow is to convert the same recording **both** ways and spike sort each, to
directly measure whether `bps=2.25` lossy compression changes sorting outcomes
vs the bit-exact `blosc-zstd` store — the gold-standard validation (see eLife
note below). Only the `traces` dataset uses the chosen codec; `times` and
properties stay on SI's lossless Blosc-zstd, so timestamps are bit-exact in both
stores.

### 3. Hybrid lossy is justified for sorting (`04_`, `05_`)

WavPack hybrid (`bps` = target bits/sample) trades a quantified error for large
gains. The only band that matters for sorting is 300–6000 Hz; this data's
in-band noise floor is **18.1 µV** (median, robust). Compression error vs that
floor:

| `bps` | ratio | 667 GB → | in-band err | err / noise floor | max abs err |
|---|---|---|---|---|---|
| 0 (lossless) | 2.05× | ~325 GB | 0 | 0 | 0 |
| 8.0 | 2.16× | ~309 GB | 0.05 µV | 0.00× | 0.8 µV |
| 6.0 | 2.66× | ~251 GB | 0.18 µV | 0.01× | 1.9 µV |
| 5.5 | 2.86× | ~233 GB | 0.27 µV | 0.01× | 2.7 µV |
| 5.0 | 3.15× | ~212 GB | 0.40 µV | 0.02× | 3.5 µV |
| 4.5 | 3.40× | ~196 GB | 0.53 µV | 0.03× | 4.9 µV |
| 4.0 | 3.91× | ~170 GB | 0.81 µV | 0.04× | 6.6 µV |
| 3.5 | 4.35× | ~153 GB | 1.10 µV | 0.06× | 9.6 µV |
| 3.0 | 5.24× | ~127 GB | 1.68 µV | 0.09× | 14 µV |
| **2.25** | **7.10×** | **~94 GB** | 3.41 µV | **0.19×** | 45 µV |

Extended bps sweep (`17_bench_bps_extended.py`, regenerates this whole table;
reproduces the original 0/2.25/3.0/3.5 rows). The high-bps end shows steep
diminishing returns: by `bps=4.0` the in-band error is already <0.05× the noise
floor, and above that the ratio falls below 4× while at `bps=8.0` it is barely
better than lossless (2.16× vs 2.05×). The compression-error/ratio trade-off does
**not** by itself say which `bps` yields acceptable *sorting* agreement — that
requires sorting at the candidate `bps` and comparing (cf. the `bps=2.25` result
in `SORTING_COMPARISON_FINDINGS.md`).

At **bps=2.25** the sorting-band error is ~5× below the noise the sorter already
contends with (effective noise √(18.1²+3.4²) = 18.4 µV, **+1.8 %**). This is the
same setting the lab uses for Neuropixel AP data
(`gfys_workspace/scripts/write_recordings.py`). Choose a higher `bps` (3.0)
for extra waveform-fidelity margin, or `bps=0` for a bit-exact archive.

> **eLife reviewed-preprint 110170** reports the **same 7.1×** at bps=2.25 (on
> Neuropixels 1.0/2.0, 30 kHz) but did **not** use a noise-floor estimate —
> they validated bps=2.25 purely through Kilosort4 sorting-performance metrics
> (accuracy/precision/recall ≈ unchanged). So our noise-floor analysis is a
> complementary, cheaper proxy; the definitive check is the planned
> sort-both-ways comparison, which mirrors their method.

### 4. Time chunk

Bigger time chunks read faster **when the consumer reads in blocks ≥ the chunk**
(t1s→t30s aligned read: 870→1821 MB/s, blosc) and produce far fewer files
(t30s/c4 ≈ 54k files/exp at ~0.5 MB each vs t1s/c4 ≈ 1.6M tiny files). The
risk is re-decode if a consumer reads windows *smaller* than the chunk (a 1 s
read against a 30 s chunk re-decodes 30×). **Mitigation:** set the sorter's
`chunk_duration` ≥ the zarr time chunk (see below). 30 s balances file count,
read speed, and memory.

## Sorting by group from the resulting store

> The full spike-sorting study that consumes these stores — lossless-vs-lossy
> agreement, MountainSort5 determinism, block/training-duration sweeps — lives in
> the sibling `../sorting/` folder (`SORTING_COMPARISON_FINDINGS.md`). This
> section only documents how the store is made sort-ready.

The store is sort-ready: `group`/`tetrode` properties and a generic-tetrode
`ProbeGroup` (synthetic geometry — relative within-tetrode position doesn't
affect per-group sorting) are baked in by `attach_tetrode_probegroup()` and
survive the zarr round-trip.

```python
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss

rec = si.read_zarr("…/2026-05-27_09-07-52.exp1.wavpack-bps2.25.zarr")
# rec already carries group/tetrode properties + a 16-tetrode ProbeGroup
groups = rec.split_by("group")                          # dict: tetrode -> 4-ch recording

# per-group preprocessing (NO common reference across tetrode channels — see SI
# tetrode guide; it can cancel spikes shared across a tetrode's 4 contacts)
pp = spre.bandpass_filter(groups, freq_min=300, freq_max=6000)
pp = spre.whiten(pp)                                    # per group

# read in blocks >= the 30 s zarr chunk so each chunk is decoded once
job_kwargs = dict(chunk_duration="30s", n_jobs=16)
sortings = ss.run_sorter_by_property(                   # or pass the dict directly
    "mountainsort5", rec, grouping_property="group",
    folder="…/sort", **job_kwargs,
)
```

Note (SI caveat): `set_times` emits a warning that per-sample times may not
propagate through some preprocessing steps. The stored `times_seg0` is intact
and `read_zarr(...).has_time_vector()` is `True`; if you need the time vector
*after* a preprocessing chain, re-apply it (e.g. carry `rec.get_times()`).

## Files

| file | purpose |
|---|---|
| `01_extract_sample.py` | pull the 180 s real sample to NVMe + time source reads |
| `02_bench_matrix.py` / `results_round1_matrix.json` | 6 compressors × 6 shapes: ratio, write, group-read |
| `03_bench_threaded_aligned.py` / `results_round2_threaded.jsonl` | production-threaded, chunk-aligned read throughput |
| `04_bench_wavpack_hybrid.py` | lossless vs `bps` 2.25/3.0/3.5/4.0: ratio + error |
| `05_bench_inband_error.py` | 300–6000 Hz error vs noise floor |
| `07_bench_chunk_shapes.py` / `results_chunk_shapes.jsonl` | channel chunk 1/4/64, size-matched; closes the chunk-size & c1 gaps |
| `17_bench_bps_extended.py` / `results_bps_ext.json` | extended `bps` sweep (0–8.0): ratio + in-band err + noise floor + max abs err; regenerates the `bps` table above |
| `06_convert.py` | **production conversion script** (`--compressor {wavpack,blosc-zstd}`) |
| `08_convert_session.py` | convert the full 2026-05-27_09-07-52 session (both experiments) to both stores |

Downstream scripts that consume these stores moved to sibling folders during the
2026-06-10 reorg: LFP generation (`09`, `13_make_subsampled_lfp`) → `../lfp/`;
spike sorting (`10`–`24`) → `../sorting/`; the loupe viewer (build → download →
launch) → `../visualization/`. See the parent `../README.md` for the full study
index.

Reproduce via the workspace env, e.g.
`cd gfys_workspace && uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/compression/02_bench_matrix.py`.
Scratch/experiment data live under `/nvme/neuropixels/tmp/cc_bench`.

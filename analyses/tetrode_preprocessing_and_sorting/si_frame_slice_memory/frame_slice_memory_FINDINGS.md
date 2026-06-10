---
title: Root cause of frame_slice + run_sorter_by_property memory blow-up (SI time-vector reload)
scope: tetrode_analyses / spikeinterface
status: active
source: measurement
created: 2026-06-07
last_updated: 2026-06-07
confidence: high
confirmed_by_user: not_required
---

# `frame_slice` + `run_sorter_by_property` memory blow-up — root cause + fix

Follow-up to `frame_slice_memory_investigation_PROMPT.md` and the
`SORTING_COMPARISON_FINDINGS.md` "Methodology note". Investigated on tononi-2,
SpikeInterface editable checkout `~/projects/ece/spikeinterface` (v0.104.1,
DEV_MODE), numpy 2.3.5.

## TL;DR

The per-worker memory does **not** scale with trace **data** — it scales with
the full-length **time vector**. `BinaryFolderRecording` persists `set_times`
to `times_cached_seg0.npy` and **eagerly `np.load`s the entire vector into
anonymous RAM on every reconstruction** (no `mmap_mode`). `run_sorter_by_property`
reconstructs the recording once per joblib worker (plus extra times within a
worker), so peak RAM ≈ (full-length float64 time vector) × (reconstructions).
The frame slice is irrelevant to this cost; no full-parent trace array is ever
allocated.

- **Root-cause line:** `spikeinterface/src/spikeinterface/core/baserecording.py`
  → `BaseRecording._extra_metadata_from_folder` →
  `time_vector = np.load(time_file)` (was line 412), reached via
  `BinaryFolderRecording.__init__` → `load_metadata_from_folder`.
- **Fix:** `np.load(time_file, mmap_mode="r")` (+ make `TimeSeries.shift_times`
  out-of-place so a read-only memmap doesn't break). Already the established
  pattern elsewhere in SI core (`npysnippetsextractor`, `node_pipeline`,
  `waveform_tools`).
- **No existing GitHub issue/PR.** Line dates to 2023-04 and was never changed
  beyond reformatting. Present in current `main`.

## Why the earlier "data × workers" attribution was wrong

The handoff inferred contribution (1) was "full parent *data* loaded per
worker" because the peak (~500 GB ≈ 4 ch × 48 h × f32 × 5 workers) matched the
per-tetrode data footprint. That match is a coincidence. Instrumenting a single
in-process run (== exactly one worker) showed:

- `numpy.load` was called **3×** per worker-run, each loading the **full** time
  vector (no `mmap_mode`), with the stack ending at `baserecording.py:412`.
- `BinaryRecordingSegment.get_traces`: **zero** big anonymous copies; max single
  read = the 300 s slice (a memmap *view*, reclaimable → not USS); total frames
  read ≈ 1210 s. The full-parent (300 M-frame) data was **never** materialized.

`get_traces` returns an `mmap` view (`binaryrecordingextractor.py:227`), which
is file-backed/reclaimable and never counts as USS, so trace data cannot
explain the anonymous-RAM peak. The 41.7 GB time vector reloaded ~2–3× per
worker × 5 workers ≈ the observed ~500 GB.

## Why it is class-specific

| class | time vector path | per-worker cost |
|---|---|---|
| `BinaryFolderRecording` (`.save`→`si.load`) | persisted to `times_cached_seg0.npy`; **reloaded eagerly** in `_extra_metadata_from_folder` on every reconstruction | full vector × reconstructions |
| `BinaryRecordingExtractor` + in-memory `set_times` | time vector lives only as a segment attribute, **not** in `_kwargs`; dropped when the recording is dumped to the sorter folder (JSON/pickle of `to_dict`) and not reloaded | none — workers reconstruct without it |

`is_binary_compatible()` is identical for both, so the ms5 wrapper's
`recording.save()` branch is taken for both — it is **not** the differentiator,
as the handoff already established.

### Why `reset_times()` on the slice did not help

`reset_times()` only nulls the *slice* segment's `time_vector`. The parent
`BinaryFolderRecording` still reloads the full vector from disk on every
reconstruction, and the `FrameSliceRecording` is rebuilt from `_kwargs`
(`parent + start + end`) in each worker, re-deriving its time vector from the
freshly-reloaded parent — so the reset is undone on reload.

## Measured before/after (cheap synthetic, tononi-2)

USS = summed `memory_full_info().uss` over the process tree.

| scenario | scale | stock | with fix |
|---|---|---|---|
| single in-process run (`diag_framesize5_instrumented.py`) | 8 ch, N=3e8, tv=2.4 GB | 5.18 GB | **0.44 GB** |
| `run_sorter_by_property` n_jobs=2 (`diag_framesize6_joblib.py`) | 8 ch, N=3e8, tv=2.4 GB | 12.66 GB | **0.87 GB** |
| `si.load(folder)` only (`si_timevector_memory_minimal_repro.py`) | 1 ch, N=2e8, tv=1.6 GB | +1.60 GB | **+0.00 GB** |

Full per-group trace data (4.8 GB at the 8 ch / N=3e8 scale) was never read in
any case. The fix bounds peak by what is actually touched (the frame slice).

## The fix (working tree only — not committed)

`core/baserecording.py` `_extra_metadata_from_folder`:
```python
time_vector = np.load(time_file, mmap_mode="r")   # was: np.load(time_file)
```
`core/time_series.py` `TimeSeries.shift_times` (read-only memmap safety):
```python
if rs.time_vector.flags.writeable:
    rs.time_vector += shift                        # in-place: no extra copy
else:
    rs.time_vector = rs.time_vector + shift        # memmap is read-only -> out-of-place
```
`np.asarray(memmap)` shares the buffer with no copy and preserves read-only, so
`TimeSeriesSegment.get_times()` stays lazy. Verified no other in-place writes to
`time_vector` in core.

Note on `shift_times` cost (measured, 1.6 GB vector): an unconditional
out-of-place add costs a transient extra full-length copy for an in-memory
writeable vector (peak +1.58 GB, settles back to 1x); the branch above avoids
that by shifting in place when possible, so the out-of-place copy is paid only
for a read-only memmap — where shifting must materialize an in-memory array
anyway (in-place is impossible). `shift_times` is a rare, explicit, top-level op
(not in the sorting hot path, not per-worker), so either form is acceptable; the
branch keeps the common case at strict 1x.

### Tests
`uv run --no-sync pytest` on `test_time_handling.py`, `test_baserecording.py`,
`test_frameslicerecording.py`, `test_channelslicerecording.py`,
`test_binaryrecordingextractor.py`, `test_recording_tools.py`,
`test_time_series_tools.py` → all pass. The save→load→`shift_times` round-trip
(`test_save_and_load_time_shift`) exercises both edits.

## Workaround without the patch

Materialize a genuine small crop and drop times before sorting (don't rely on
`frame_slice` + `reset_times` of a large `BinaryFolderRecording`):
`rec.frame_slice(0, T).save(folder=crop, ...)` then `si.load(crop).reset_times()`
— the cached crop's `times_cached` file is short, so the per-worker reload is
cheap. (This matches the per-crop benchmarking fix already noted in
`SORTING_COMPARISON_FINDINGS.md`.)

## Upstream issue (draft)

**Title:** `BinaryFolderRecording` eagerly loads full cached time vector into
RAM on every reconstruction → `run_sorter_by_property` memory scales with
recording length × n_workers.

**Body:** `_extra_metadata_from_folder` does `np.load(times_cached_seg0.npy)`
without `mmap_mode`, so reconstructing a `BinaryFolderRecording` pulls the entire
float64 time vector (1 value/sample; tens of GB for long recordings) into
anonymous RAM, even for a short `frame_slice`. `run_sorter_by_property`
reconstructs per joblib worker → peak ≈ time_vector × n_workers. Trace data is
never the cause (`get_traces` returns mmap views). Minimal repro:
`si_timevector_memory_minimal_repro.py` (sorter-free; prints `+1.60 GB` on
`si.load` for a 1-channel recording needing only a 1 s slice). Proposed fix:
`np.load(time_file, mmap_mode="r")` (already used for `.npy` loads elsewhere in
core) + make `shift_times` out-of-place.

## Scripts
`tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/si_frame_slice_memory/`:
`diag_framesize5_instrumented.py` (localization), `diag_framesize6_joblib.py`
(before/after harness), `si_timevector_memory_minimal_repro.py` (filing repro).

# Agent handoff: root-cause the `frame_slice` + `run_sorter_by_property` memory inefficiency

Self-contained prompt for another agent. Background: surfaced while benchmarking
MountainSort5 `scheme2_training_duration_sec` footprint on tononi-2 (see
`SORTING_COMPARISON_FINDINGS.md` "Methodology note" and `diag_framesize*.py`).

---

**Task: Root-cause and (if possible) eliminate a memory inefficiency in the SpikeInterface `frame_slice` + `run_sorter_by_property` + `BinaryFolderRecording` path.**

**Context / the phenomenon.** On tononi-2, sorting tetrode data with MountainSort5 via SpikeInterface, I found that when the input recording is a `frame_slice(0, T)` of a *large* `BinaryFolderRecording` (the class produced by `recording.save(format="binary")` then `si.load(...)`), each worker's peak memory scales with the **full parent recording length, not the frame-sliced length T**. The sort's *output* is correct (it processes only T seconds and yields ~T-worth of units), so `frame_slice`'s length is honored — only memory is full-parent-scaled. Concretely: `frame_slice(0, 300 s)` of a 48 h (5.2×10⁹ frames) × 64 ch float32 binary, sorted via `run_sorter_by_property(grouping_property="group", n_jobs=5)`, peaked at **~500 GB** ≈ full-per-tetrode (4 ch × 48 h × f32 ≈ 83 GB) × 5 workers.

**What I've already established** (staged synthetic reproductions; measure memory as **summed USS** — `psutil.Process().memory_full_info().uss` over the process tree, NOT RSS, which counts reclaimable mmap'd file cache):

| recording class | sorter call | scale | peak USS |
|---|---|---|---|
| `BinaryRecordingExtractor` (single `.dat`) | `run_sorter` | 4 ch, 2.2×10⁹ frames | 0.25 GB |
| `BinaryRecordingExtractor` | `run_sorter_by_property` | 64 ch, 5.2×10⁹ frames | 1.76 GB |
| `BinaryRecordingExtractor` + in-memory `set_times` (full float64 vector) | `run_sorter_by_property` | 64 ch, 5.2×10⁹ | 43 GB (≈ the 41.7 GB time vector) |
| **`BinaryFolderRecording`** (`.save`→`si.load`) | `run_sorter_by_property` | 8 ch, 1.5×10⁹ | 60–73 GB (≈ full per-group data × workers; persists under `reset_times`) |

Pinned facts:
- **The decisive variable is the recording class:** `BinaryFolderRecording` exhibits it; `BinaryRecordingExtractor` does not (even at the exact 48 h frame count).
- Two contributions: **(1, dominant)** full parent *data* loaded per worker in the `BinaryFolderRecording` path — confirmed because it persists when the time vector is dropped via `reset_times()`; **(2, secondary)** a full-length float64 time vector that `frame_slice` carries at parent length (~42 GB).
- `is_binary_compatible()` is **identical** for both classes (`True` for the full recording, `False` for a `frame_slice`), so the ms5 wrapper branch `if not recording.is_binary_compatible(): recording_cached = recording.save(...)` is taken for *both* — i.e. that branch alone is **not** the differentiator.

**What is NOT yet known (your goal):** the *exact* code path/line where the full parent gets read into worker memory despite the frame slice, why `BinaryFolderRecording` differs from `BinaryRecordingExtractor`, and whether it can be eliminated.

**Investigation steps:**
1. Reproduce cheaply using sparse synthetic binaries (truncate a file to size, write noise only to the first ~320 s) at 4–8 channels so crossing large frame counts costs ~tens of GB, not 1.3 TB. Starting-point scripts exist: `~/projects/ece/tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/si_frame_slice_memory/diag_framesize{1,2,3,4}.py` — `diag_framesize4.py` is the one that reproduces it.
2. Instrument to localize the allocation: candidates are the ms5 SI wrapper's `recording.save()` (`spikeinterface/src/spikeinterface/sorters/external/mountainsort5.py` ~line 187), `run_sorter_by_property` / `split_by` / the channel-slice-of-frame-slice serialization to joblib workers, `BinaryFolderRecording` vs `BinaryRecordingExtractor` `frame_slice`/`get_traces`, and ms5's `get_sampled_recording_for_training` (`~/projects/ece/mountainsort5/mountainsort5/schemes/`). Use `tracemalloc`, per-line memory profiling, and/or printing array sizes/`get_num_samples()` at each hop. Pin the file:line where a full-parent-sized array is allocated.
3. Determine why the two recording classes diverge despite identical `is_binary_compatible()`.
4. Assess elimination: a usable workaround (e.g., materialize a genuine crop; `reset_times()`; a different way to pass the recording) **and** the ideal upstream fix (a small SI patch so worker memory is bounded by the frame slice). If you propose a patch, verify it on the diag4 repro (USS should drop to the genuine-crop level) and run relevant SI core tests (`cd ~/projects/ece/gfys_workspace && uv run --no-sync pytest ../spikeinterface/src/spikeinterface/core/tests/...`).
5. **Search existing GitHub issues/PRs** in `SpikeInterface/spikeinterface` (use `gh issue list`/`gh search issues --repo SpikeInterface/spikeinterface` and the web) for prior reports — try terms like `frame_slice memory`, `run_sorter_by_property memory`, `split_by memory`, `BinaryFolderRecording frame_slice`, `frame slice loads full recording`, `recording.save sorter memory`. Link anything relevant and note whether it's fixed in a newer version.

**Environment rules:** Run Python via `cd ~/projects/ece/gfys_workspace && uv run --no-sync python ...` — **never** `uv sync`. The SI checkout is editable at `~/projects/ece/spikeinterface`; ms5 at `~/projects/ece/mountainsort5`. tononi-2 is a **shared 1.5 TiB host** — be courteous, keep peak memory modest, scope any process management to the current user, use `/nvme` for scratch and clean it up.

**Deliverable:** (a) the exact root-cause location (file:line / call path) of the full-parent memory load and why it's class-specific; (b) whether it can be eliminated, with a verified workaround and a proposed upstream fix if feasible; (c) results of the GitHub issue search (links + fixed-in-version status); (d) a minimal (<50-line) standalone reproduction suitable for filing an SI issue. Report findings; do not commit or push.

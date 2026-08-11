---
title: Kilosort4 feasibility for the full-48h tetrode sort — the GPU clustering memory wall
updated: 2026-06-12
---

# Kilosort4 on the full 48 h tetrode session: feasibility and the GPU clustering wall

## Conclusion

**Kilosort4 cannot complete a sort of the full 48 h tetrode session
(`2026-05-27_09-07-52`; 64 ch = 16 tetrodes × 4 ch, 30 kHz, 5.215e9 frames) on the
host's 32 GB Tesla V100** — whether sorting all tetrodes **together** (one 64-ch run)
or **by tetrode group** (one 4-ch run per tetrode). The blocker is KS4's final
clustering step, which allocates an `(n_spikes × n_clusters)` **dense** tensor on the
GPU. At 48 h a single tetrode has ~24 M spikes, so that allocation (~12.6 GiB) on top
of the ~30 GiB clustering working set needs **~42 GiB — over the 32 GB card**. This is
**independent of the together-vs-by-group layout and of channel count** (a single 4-ch
tetrode hits the same wall as the full 64-ch run) and **no KS4 parameter chunks it**.

Materialize + detection work and are GPU-accelerated; only the clustering does not fit.
MountainSort5, by contrast, sorts the same 48 h in ~100 min because its clustering runs
on the CPU with the host's 1.5 TB RAM. **For 48 h tetrode data on a 32 GB GPU, MS5 is
feasible and KS4 is not.**

## What works

- **Install** (Python 3.14 / cp314 wheels, V100): `kilosort>=4.1.7`, `torch==2.9.1`
  (pinned to force a cu128 build with sm_70/Volta — uv's auto torch-backend otherwise
  grabs cu130, whose wheels drop sm_70 → "no kernel image" on the V100),
  `faiss-cpu==1.12.0` (only release with a cp314 wheel), numba 0.65.1. Verified
  `torch.cuda` runs on the V100. See `[[project_ks4_tetrode_sorting]]` (agent memory).
- **Preprocessing materialize** (bandpass 300–6000 Hz + global CMR, float32, 48 h):
  34.9 min, peak USS 84.3 GB, peak /nvme 1379 GB — **identical to MS5's materialize**
  (the cache is byte-for-byte the same; both sorters consume it).
- **Smoke (600 s crop)**: both layouts produce sensible per-tetrode units in seconds
  (by-group 124 units; together 116–135 units across all 16 tetrodes).
- **KS4 detection at 48 h (GPU)**: ~62 min/tetrode (by-group, 4 ch), 15.8 M spikes
  detected on tetrode 0; GPU at 90–100 % util. The detection stage scales fine.

## The failure ladder (each fix revealed the next)

1. **torch auto-backend → CUDA 13 (cu130)**, whose wheels are built for sm_75+ only →
   `no kernel image is available` on the V100 (sm_70). **Fix:** pin `torch==2.9.1`
   (resolves to `+cu128`, arch list includes sm_70).
2. **faiss-cpu** latest (1.13/1.14) has no cp314 wheel. **Fix:** pin `faiss-cpu==1.12.0`.
3. **`templates_from_data=True` (default) hangs for hours.** `spikedetect.extract_wPCA_wTEMP`
   loops sampled batches across the WHOLE recording collecting threshold-crossing clips
   with **no cap**, then `TruncatedSVD` + `KMeans(n_init=10)` on them; at 48 h × 64 ch
   that is millions of clips and the KMeans churns for hours (futex-blocked, GPU idle,
   I/O frozen — looks like a hang). **Fix:** `templates_from_data=False` (KS4's prefab
   universal templates seed detection; real templates are still learned during clustering).
4. **together (64 ch): OOM at the final clustering.** Full pipeline ran ~6 h (detection
   1.6 h, clustering 2.1 h over 16 spatial centers, final extraction 1.5 h) then
   `torch.OutOfMemoryError` at `clustering_qr.assign_iclust`: 29.45 GiB allocated +
   12.34 GiB requested.
5. **by-group (4 ch/tetrode): OOM at the same step, on the FIRST tetrode.** Tetrode 0:
   detection 62 min (15.8 M spikes) → clustering 16 min (16 clusters) → final extraction
   4.5 min (23.6 M spikes) → OOM at `assign_iclust`: **29.98 GiB allocated + 12.57 GiB
   requested**. Nearly identical to the 64-ch together run → the wall is the **per-tetrode
   48 h spike volume**, not the channel count or the joint-clustering of tetrodes.

## Root cause (code-level)

`kilosort/clustering_qr.py::assign_iclust` (called `niter` times per center):

```python
xN = coo(ij, tones2.flatten(), (n_spikes, nclust)).to_dense()   # (n_spikes × nclust) DENSE
...
xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense())            # broadcasts to (n_spikes × nclust)
```

With `n_spikes ≈ 23.6 M` and `nclust ≈ 130–200`, `(n_spikes × nclust)` float32 ≈ 12.6 GiB.
The ~30 GiB working set is the spike features `Xg` plus `kn`, `rows_neigh`, `tones2` — all
`∝ n_spikes`. Total ≈ ~42 GiB.

- `max_cluster_subset` (default 25 000) and `cluster_downsampling` (default 20) bound
  only the **landmark** subset (`n_nodes` / `nsub`), **not** `n_spikes` (the rows of
  `xN`), so they do not shrink this allocation.
- `expandable_segments` confirmed it is **not fragmentation** (reserved-but-unallocated
  fell to ~46–85 MiB; the deficit is genuine allocated memory). Note the env name
  `PYTORCH_CUDA_ALLOC_CONF` is deprecated in torch 2.9 (`PYTORCH_ALLOC_CONF`), but it is
  moot — no allocator setting overcomes a real 42 GiB requirement on a 32 GB card.

## Scaling — what would fit

`n_spikes ∝ duration`. ~42 GiB at 48 h ⇒ the V100 fits roughly **≤ ~28 h per tetrode**.
Paths to a KS4 result on this hardware, **not pursued** (user chose to stop & document):

- sort a ≤ 24 h window (≈12 M spikes/tetrode → fits);
- raise detection thresholds to cut spikes ~40 % so 48 h fits (changes the sort vs MS5;
  needs tuning runs);
- time-block + stitch units across blocks (overlaps the `34_track` unit-tracking project);
- run clustering on CPU (`torch_device="cpu"`, uses the 1.5 TB RAM but infeasibly slow);
- a larger GPU (≥ 48 GB) would clear 48 h directly.

## Compute footprint vs MS5 (what was measured)

| stage | MS5 (scheme-2 single block, `27_`) | KS4 (by-group, no drift, `30_`) |
|---|---|---|
| materialize (bandpass + global CMR, f32) | 32.7 min · USS 83.5 GB · /nvme 1377 GB | 34.9 min · USS 84.3 GB · /nvme 1379 GB (identical) |
| sort | **99.8 min**, CPU (n_jobs=5), no GPU, peak USS 81.8 GB | **did not complete** — detection ~62 min/tetrode on GPU; clustering OOMs |
| compute device | CPU + 1.5 TB RAM | GPU 32 GB (the binding limit) |
| total units (48 h) | 106 | n/a |

## Code & artifacts

- `30_sort_ks4.py` — KS4 sort driver (by tetrode group, no drift, `templates_from_data=False`).
  Runs end-to-end through detection but OOMs at the 48 h clustering. Reusable as-is for a
  ≤ 24 h window or on a larger GPU.
- `31_build_analyzer_ks4.py`, `32_compare_ks4_vs_ms5.py` — downstream analyzer + KS4-vs-MS5
  agreement, staged and linted; ready to run if/when a completed KS4 sort exists.
- `tetrode_analyses.sorting`: `sort_store_ks4` (by-group or together; the
  `templates_from_data`/`clear_cache` knobs and the per-tetrode `.dat` rewrite parallelism
  live here), `materialize_preprocessed` (shared with MS5's `sort_store`),
  `assign_tetrode_groups` (per-unit tetrode from KS4's own `get_best_channels`, for the
  together layout).

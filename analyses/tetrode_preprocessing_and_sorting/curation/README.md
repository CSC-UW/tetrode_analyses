---
title: Spike-sorting curation (server + local) for session 2026-05-27_09-07-52
scope: tetrode_analyses
status: active
source: code_inspection
created: 2026-06-11
last_updated: 2026-06-12
confidence: high
confirmed_by_user: not_required
---

# Curation

Curate the chunk-tracked MountainSort5 sorting (`tracked_48h/analyzer_clustered.zarr`,
scripts 36/37) in [spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui)
(the `grahamfindlay/spikeinterface-gui@dev` fork, via the `curation` extra), two ways.
Part of the `tetrode_preprocessing_and_sorting` study (see `../README.md`).

## Files

| file | runs on | purpose |
|---|---|---|
| `launch_curation.py` | **tononi-2** | Curate directly on the server (analyzer + recording already local). Web (SSH-tunnel) or desktop (`ssh -X`). |
| `build_curation_manifest.py` | **tononi-2** | Write a lightweight JSON manifest of the items needed to curate locally + their server paths. |
| `download_curation_bundle.py` | **local** | rclone the manifest + its items to an external drive. |
| `launch_curation_local.py` | **local** | Curate over the downloaded bundle (Qt desktop or web). |
| `grahams_curation_layout.json`, `grahams_curation_settings.json` | both | `--style grahams_curation` preset (layout + per-view settings). |

## Local curation pipeline

```bash
# 1. On tononi-2 — write the manifest (no data copied):
cd gfys_workspace
uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/curation/build_curation_manifest.py
#   ... or --no-traces for an analyzer-only (no raw traces) manifest.

# 2. On your local machine (rclone "tononi-2" remote, rooted at filesystem /):
python download_curation_bundle.py --dest /Volumes/MyDrive/ttm_nod_curation
#   ~355 GB with traces; ~2.5 GB with --no-traces.

# 3. On your local machine, in a tetrode_analyses checkout with the curation extra:
uv sync --extra curation
uv run python launch_curation_local.py --data-dir /Volumes/MyDrive/ttm_nod_curation --style grahams_curation
```

## Why a manifest instead of a bundle folder

Unlike the loupe viz bundle, curation data is too large to stage into one folder:
the recording is hundreds of GB. So the "bundle" is a **manifest** listing the
items + their tononi-2 locations, and the download rclones each from its real
location. Curation needs only:

- `analyzer_clustered.zarr` (~2.5 GB) — the SortingAnalyzer; it **embeds its own
  sorting** (so `aggregated/`/`by_group/` are not downloaded) and references its recording.
- the `blosc-zstd` recording (~352 GB) — **only for the traces view**, and only
  the store the sort was actually produced from (no lossy/alternative variants).

## Recording auto-resolution & `--no-traces`

`analyzer_clustered.zarr` references its recording by a *relative* path
(`../../../<session>.blosc-zstd.zarr`). The download preserves that layout —
analyzer at `sortings_seed42_pcafix/tracked_48h/analyzer_clustered.zarr`, recording at
the drive root — so `load_sorting_analyzer` **auto-resolves** the recording with zero
reconstruction (bit-exact traces). `launch_curation_local.py` then just checks
`analyzer.has_recording()`.

`--no-traces` (at build, download, and launch) omits the recording and runs
`run_mainwindow(..., with_traces=False)`: you still get the unit list, metrics,
waveforms, templates, correlograms, and curation — everything except the
raw-traces view.

Default target: `tracked_48h` / `analyzer_clustered.zarr` — the chunk-tracked 48 h sort
(scripts 36/37; the recommended sorting, geometry-free QC; well-isolated tiers
262 permissive / 173 moderate / 103 conservative). Override with `--sorting` /
`--analyzer-name` (e.g. the older 12 h-block sort: `--sorting blosc-43200s-train3600s
--analyzer-name analyzer.zarr`).

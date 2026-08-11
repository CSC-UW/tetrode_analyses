---
title: Local loupe visualization of tetrode session 2026-05-27_09-07-52
updated: 2026-06-11
---

# Visualization

Review the full 48 h TTM-001 / TTM-NOD session — synthetic EMG, sub-LFP, and the
MountainSort5 sorting — in a local [`loupe`](https://github.com/CSC-UW/loupe)
viewer, all on one session-relative clock. Part of the
`tetrode_preprocessing_and_sorting` study (see `../README.md`).

The data live on tononi-2; the viewer runs on your laptop. The split is three
scripts: **build** a portable bundle on the server, **download** it to an
external drive, then **launch** the viewer over that drive.

## Files

| file | runs on | purpose |
|---|---|---|
| `build_bundle.py` | **tononi-2** (workspace env) | Assemble `<session>/viz_bundle/`: copy the EMG zarr, re-export the 125 Hz sub-LFP as a plain xarray zarr, convert the sorting to `spikes.parquet` (session-relative seconds), write `manifest.json`. |
| `download_bundle.py` | **local** (stdlib + `rclone`) | `rclone copy tononi-2:<bundle> <external-drive>` with parallel transfers + progress. |
| `launch_loupe.py` | **local** (`tetrode_analyses[viz]`) | SpikeInterface-free loupe viewer over the downloaded bundle. |

## Run order

```bash
# 1. On tononi-2 (builds the 125 Hz LFP if missing; emits ~3 GB bundle):
cd gfys_workspace
uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/visualization/build_bundle.py

# 2. On your local machine (rclone "tononi-2" remote, rooted at filesystem /):
python download_bundle.py --dest /Volumes/MyDrive/ttm_nod

# 3. On your local machine, in a tetrode_analyses checkout with the viz extra:
uv sync --extra viz
uv run python launch_loupe.py --data-dir /Volumes/MyDrive/ttm_nod
```

## Bundle contents

```
viz_bundle/
  synthetic_emg_methods.zarr   # copied verbatim (already a plain xarray zarr)
  lfp.125hz.zarr               # 16 tetrode-lead sub-LFP, re-exported as plain xarray
  spikes.parquet               # one row per spike: time[s, float64], unit_id, tetrode
  state_definitions.json       # loupe scoring keymap + label colors (from sleepscore/launch_scoring)
  manifest.json                # session/sorting provenance + per-tetrode separator boundaries
```

`state_definitions.json` (versioned next to the scripts, copied verbatim into the
bundle) carries the sleep-scoring keymap + per-state colors from
`cnpix/sleepscore/launch_scoring.py`; `launch_loupe.py` passes it to
`lp.view(state_definitions=...)`, which enables loupe's interval scoring (without
it, loupe raises `LoupeConfigError: No state definitions found`).

## Why a server-side prep step

Two things the viewer needs aren't directly downloadable:

- **125 Hz sub-LFP** — only the 625 Hz LFP exists on disk, and a
  SpikeInterface-saved zarr is not directly `xr.open_zarr`-able. `build_bundle.py`
  generates the sub-LFP (via `tetrode_analyses.lfp.make_subsampled_lfp`, the same
  call as `../lfp/13_make_subsampled_lfp.py`) and re-exports it as a plain xarray
  zarr so the viewer reads it without SpikeInterface.
- **Spike times in seconds** — mapping spike frames → session-relative seconds
  needs the source store's time vector, which is buried inside a multi-hundred-GB
  zarr. We do that once on the fast box (SpikeInterface's native
  `get_unit_spike_train(return_times=True)`) and emit a compact parquet, rather
  than shipping the source store.

The payoff: the local install is just `tetrode_analyses[viz]` (base deps +
`loupe`) — **no SpikeInterface, ecephys, mountainsort5, source recording, or
NFS**. The viewer imports `tetrode_analyses.viz` (the Open Ephys tetrode palette)
which pulls no SpikeInterface.

## Panes & the tetrode separators

Top to bottom: synthetic EMG (both estimators), the 16 dense sub-LFP leads, and
the spike raster. The raster is a **single pane**, colored by tetrode, with thin
**horizontal separators between tetrodes** — loupe's `horizontal_separators`
(unit_id boundaries computed from the parquet; falls back to `split_by="tetrode"`
if unit_ids aren't tetrode-contiguous). LFP and raster share the 16-color Open
Ephys "Classic" palette (`tetrode_analyses.viz.tetrode_color_map`).

By default the bundle targets the 48 h / 12 h-block / 1 h-training-window sorting
(`blosc-43200s-train3600s`); override with `build_bundle.py --sorting <name>`.

---
title: LFP generation for tetrode session 2026-05-27_09-07-52
updated: 2026-06-11
---

# LFP

Downstream of the compressed Zarr stores (`../compression/`): build LFPs from the
30 kHz store. Part of the `tetrode_preprocessing_and_sorting` study (see
`../README.md`). The interactive **loupe** viewer that consumes these LFPs moved
to [`../visualization/`](../visualization/README.md) (`launch_loupe.py`).

## Files

| file | purpose |
|---|---|
| `09_make_lfps.py` | 30 kHz store → **625 Hz** full-probe LFP (`make_lfp`) + verification plot (`lfp_full_session_check.png`) |
| `13_make_subsampled_lfp.py` | 625 Hz LFP → **125 Hz** 16 tetrode-lead LFP (`make_subsampled_lfp`, ÷5 anti-aliased `resample`) → `*.lfp.125hz.zarr` + verification plot (`lfp_downsample_check.png`) |

## Subsampled LFP + viewer

`13_make_subsampled_lfp.py` keeps one lead channel per tetrode (every 4th
channel — overridable via `make_subsampled_lfp(..., channel_ids=[...])`) and
resamples the 625 Hz LFP to 125 Hz. It reuses `make_lfp`'s `resample` step: an
**integer** ratio (625/125 = 5) makes `resample` dispatch to the anti-aliased
`scipy.signal.decimate` instead of its FFT fallback (which SpikeInterface warns
against for non-integer ratios — it assumes periodicity and rings at the edges).
`resample_rate` must divide the parent rate, so 125 Hz (or 625/5) is the natural
target; a clean ÷6 → 104.17 Hz isn't an integer rate and is rejected. The store
matches `make_lfp`'s conventions, so `open_lfps_dataarray` reads it unchanged
(dims `(time, channel)`; `time`/`channel`/`tetrode` coords; `fs`/`units` attrs).

The 125 Hz sub-LFP is what the loupe viewer's LFP pane draws; building it here
(rather than on demand) lets `../visualization/build_bundle.py` re-export it as a
plain xarray zarr for SpikeInterface-free local viewing.

Reproduce via the workspace env, e.g.
`cd gfys_workspace && uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/lfp/09_make_lfps.py`.

---
title: LFP generation + loupe viewer for tetrode session 2026-05-27_09-07-52
scope: tetrode_analyses
status: active
source: code_inspection
created: 2026-05-31
last_updated: 2026-06-10
confidence: high
confirmed_by_user: not_required
---

# LFP + viewer

Downstream of the compressed Zarr stores (`../compression/`): build LFPs from the
30 kHz store and launch an interactive viewer. Part of the
`tetrode_preprocessing_and_sorting` study (see `../README.md`).

## Files

| file | purpose |
|---|---|
| `09_make_lfps.py` | 30 kHz store → **625 Hz** full-probe LFP (`make_lfp`) + verification plot (`lfp_full_session_check.png`) |
| `13_make_subsampled_lfp.py` | 625 Hz LFP → **125 Hz** 16 tetrode-lead LFP (`make_subsampled_lfp`, ÷5 anti-aliased `resample`) → `*.lfp.125hz.zarr` + verification plot (`lfp_downsample_check.png`) |
| `14_launch_loupe.py` | launch a **loupe** viewer: synthetic EMG (per_window + global) on top, then 16 sub-LFP traces (dense, colored by tetrode) over a spike raster split/colored by tetrode; `--sorting {blosc-zstd,wavpack-bps2.25}` (default `blosc-zstd`) |

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

`14_launch_loupe.py` stacks three panes: the synthetic EMG (both estimators,
loaded from `synthetic_emg_methods.zarr`; see `analyses/emg/`), the 16 dense LFP
traces, and the spike raster. LFP traces and the raster are colored by tetrode
with the 16-color Open Ephys "Classic" channel palette (`tetrode_analyses.viz`).
Spike times come straight from SpikeInterface: the 30 kHz source recording (which
carries the session-relative time vector) is registered to the sorting, and
`get_unit_spike_train(return_times=True)` converts frames to seconds — so EMG,
LFP, and spikes share one clock.

Reproduce via the workspace env, e.g.
`cd gfys_workspace && uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/lfp/09_make_lfps.py`.

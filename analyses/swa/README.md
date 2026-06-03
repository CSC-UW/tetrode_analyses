# SWA (slow-wave activity / delta power) — TTM-001 / TTM-NOD

Slow-wave activity across the whole tetrode recording, with the light/dark cycle
and the day-2 sleep-deprivation (novel-objects) window drawn underneath — the
tetrode analogue of the SWA timetraces in `offproj` / `findlay2025a`.

Two independent SWA estimates are produced, both averaged **per tetrode** (mean
over each tetrode's 4 channels; ~4x smaller on disk than per-channel) and with
**no bipolar referencing**:

- **Instantaneous delta power** — bandpass (0.5–4 Hz) + Hilbert envelope, decimated
  to 125 Hz. Follows `wisc_ecephys_tools.rats.pipeline.get_instantaneous_power`.
- **STFT band power** — 4 s DPSS STFT, q=2 decimation (312.5 Hz), power summed per
  band. Follows `findlay2025a.pipeline.compute_cx_bandpowers_and_psds` (computes
  delta, vlad/eta, theta, sigma, gamma; only delta is plotted here).

Reusable library code lives in the package, not in these scripts:
`tetrode_analyses.experiment` (metadata + datetime↔session-time mapping + light/dark
periods), `tetrode_analyses.power` (band-power extraction), and
`tetrode_analyses.plotting` (SWA timetraces + overlays).

## Inputs

- LFP: `/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/2026-05-27_09-07-52.lfp.zarr`
  (625 Hz, 64 ch = 16 tetrodes, ~48.3 h; session-relative `time` coord).
- Session anchor: first segment `software_time_s` from the session `slice_table.csv`
  (2026-05-27 09:07:52 America/Chicago).

## Outputs (`/nvme/neuropixels/tetrode_data/TTM-001/TTM-NOD/`)

- `experiment_params.json` — light/dark cycle, deprivation window, session anchor.
- `delta.idelta.zarr` — per-tetrode instantaneous delta power `(tetrode, time)`.
- `stft_bandpowers.zarr` — per-tetrode STFT band powers, one `(tetrode, time)` var/band.
- `figures/swa_{inst,stft}_{mean,per_tetrode}.png` (also copied next to the scripts).

## Run (from `gfys_workspace/`)

```bash
uv run ../tetrode_analyses/analyses/swa/00_write_experiment_params.py
uv run ../tetrode_analyses/analyses/swa/01_extract_inst_delta.py
uv run ../tetrode_analyses/analyses/swa/02_compute_stft_bandpowers.py
uv run ../tetrode_analyses/analyses/swa/03_plot_swa.py
```

## Notes

- No hypnogram exists for this subject yet, so (unlike `findlay2025a`) per-condition
  PSDs and hypnogram masking are out of scope; only full-recording timeseries are saved.
- Artifact transients are removed at **plot time** per tetrode via
  `power.replace_outliers` (the histogram-gap method ported from
  `findlay2025a`); the saved `.zarr` products stay raw. Smoothing uses
  `min_periods=1` so the NaN'd outliers are skipped rather than blanking windows.
- The recording concatenates two Open Ephys experiments with a ~58 s wall-clock gap;
  `xrsig.stft_psd` segments on that gap automatically. The instantaneous-power path
  filters across it (one boundary in ~48 h; negligible for the delta envelope), as the
  rat reference pipeline does.

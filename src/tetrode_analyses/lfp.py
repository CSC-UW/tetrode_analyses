"""LFP production and loading for tetrode Zarr stores.

`make_lfp` downsamples a converted 30 kHz tetrode Zarr store to a 625 Hz LFP
Zarr store (float32, microvolts) using a two-stage anti-aliased decimation
(SpikeInterface `resample`, which applies `scipy.signal.decimate`). The
session-relative time vector is carried through (subsampled) and stored
Delta-compressed.

`make_subsampled_lfp` takes an existing LFP store, keeps a subset of channels
(by default one lead channel per tetrode, via `tetrode_lead_channel_ids`), and
downsamples it further to a lower rate — e.g. 625 Hz -> 125 Hz (÷5) on the 16
tetrode leads — producing a lighter store for fast whole-session review. It uses
the same anti-aliased decimation as `make_lfp` (`resample` with an integer ratio,
which dispatches to `scipy.signal.decimate`). Output uses the same store
conventions as `make_lfp`, so it is readable by `open_lfps_dataarray`.

`open_lfps_dataarray` opens an LFP store as a lazy, dask-backed xarray
DataArray, matching the coordinate conventions of
`tetrode_analyses.io.open_tetrode_dataarray`.
"""

from __future__ import annotations

import pathlib

import dask.array as da
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import xarray as xr
import zarr
from numcodecs import Delta

# 30 kHz -> 625 Hz is a decimation factor of 48. scipy.signal.decimate degrades
# for a single large factor (recommend q <= ~13), so cascade two integer stages.
# 8 then 6: each stage's Chebyshev IIR stays low-order, and the final Nyquist
# (312.5 Hz) sits well below each intermediate cutoff.
DEFAULT_STAGE_RATES = (3750, 625)  # 30000/8=3750, 3750/6=625


def downsample_to_lfp(
    recording,
    *,
    stage_rates: tuple[int, ...] = DEFAULT_STAGE_RATES,
    dtype: str = "float32",
):
    """Return a lazily-resampled LFP recording (microvolts) at ``stage_rates[-1]``.

    Scales int traces to microvolts first, then applies anti-aliased decimation
    in stages. Each ``parent_rate / stage_rate`` must be an integer so
    `resample` uses `scipy.signal.decimate` (not the FFT fallback).
    """
    rec = spre.scale_to_uV(recording) if recording.has_scaleable_traces() else recording
    for rate in stage_rates:
        parent = rec.get_sampling_frequency()
        if parent % rate != 0:
            raise ValueError(
                f"Stage rate {rate} does not evenly divide parent rate {parent}; "
                "pick integer-ratio stages so decimate (anti-aliased) is used."
            )
        rec = spre.resample(rec, rate, dtype=dtype)
    return rec


def make_lfp(
    in_zarr: str | pathlib.Path,
    out_zarr: str | pathlib.Path,
    *,
    stage_rates: tuple[int, ...] = DEFAULT_STAGE_RATES,
    time_chunk_s: float = 300.0,
    n_jobs: int = 16,
) -> object:
    """Produce a 625 Hz float32 LFP Zarr store from a 30 kHz tetrode store."""
    out_zarr = pathlib.Path(out_zarr)
    rec = si.read_zarr(str(in_zarr))
    lfp = downsample_to_lfp(rec, stage_rates=stage_rates)
    print(
        f"LFP: {rec.get_sampling_frequency():.0f} Hz -> {lfp.get_sampling_frequency():.0f} Hz "
        f"({rec.get_num_channels()} ch, {lfp.get_num_frames() / lfp.get_sampling_frequency() / 3600:.2f} h) "
        f"-> {out_zarr}"
    )
    return lfp.save(
        format="zarr",
        folder=str(out_zarr),
        filters_by_dataset={"times": [Delta(dtype="float64")]},
        chunk_duration=f"{time_chunk_s}s",  # all channels per chunk (LFP read whole-probe)
        n_jobs=n_jobs,
        progress_bar=True,
    )


def tetrode_lead_channel_ids(recording) -> list:
    """Return the channel id of the first ("lead") channel of each tetrode.

    Groups the recording's channels by the 0-based ``group`` property (set by
    :func:`tetrode_analyses.io.attach_tetrode_probegroup`) and returns the first
    channel id of each group in ascending group order — i.e. every 4th channel
    when channels are in tetrode-contiguous order. Falls back to the ``tetrode``
    property (first-appearance order) when ``group`` is absent.
    """
    prop_keys = recording.get_property_keys()
    channel_ids = list(recording.get_channel_ids())
    if "group" in prop_keys:
        key = np.asarray(recording.get_property("group"))
        order = sorted(set(key.tolist()))
    elif "tetrode" in prop_keys:
        key = np.asarray(recording.get_property("tetrode"))
        order = list(dict.fromkeys(key.tolist()))  # first-appearance order
    else:
        raise ValueError(
            "recording has no 'group' or 'tetrode' property; pass channel_ids explicitly."
        )
    return [channel_ids[int(np.flatnonzero(key == g)[0])] for g in order]


def make_subsampled_lfp(
    in_lfp_zarr: str | pathlib.Path,
    out_zarr: str | pathlib.Path,
    *,
    channel_ids: list | None = None,
    resample_rate: int = 125,
    time_chunk_s: float = 300.0,
    n_jobs: int = 16,
) -> object:
    """Downsample selected channels of an LFP Zarr store to ``resample_rate`` Hz.

    Keeps ``channel_ids`` (default: one lead channel per tetrode, via
    :func:`tetrode_lead_channel_ids`) and resamples with `resample` exactly as
    :func:`make_lfp` does. ``resample_rate`` must evenly divide the parent rate so
    `resample` dispatches to the anti-aliased `scipy.signal.decimate` (rather than
    its FFT fallback, which is what SI warns against for non-integer ratios).
    Defaults take the 625 Hz LFP's 16 tetrode leads to 125 Hz (÷5). The
    session-relative time vector is carried through (decimated) and the store
    matches :func:`make_lfp`'s conventions, so :func:`open_lfps_dataarray` reads
    it unchanged.
    """
    out_zarr = pathlib.Path(out_zarr)
    rec = si.load(str(in_lfp_zarr))
    ids = list(channel_ids) if channel_ids is not None else tetrode_lead_channel_ids(rec)
    sub = rec.select_channels(ids)

    parent_fs = sub.get_sampling_frequency()
    if parent_fs % resample_rate != 0:
        raise ValueError(
            f"resample_rate {resample_rate} does not evenly divide parent rate "
            f"{parent_fs}; pick a divisor (e.g. 125 = 625/5) so resample uses the "
            "anti-aliased scipy.signal.decimate rather than its FFT fallback."
        )
    lfp = spre.resample(sub, resample_rate)
    print(
        f"sub-LFP: {parent_fs:.2f} Hz -> {resample_rate} Hz (÷{int(parent_fs / resample_rate)}) | "
        f"{sub.get_num_channels()} ch, "
        f"{lfp.get_num_frames() / resample_rate / 3600:.2f} h -> {out_zarr}"
    )
    return lfp.save(
        format="zarr",
        folder=str(out_zarr),
        filters_by_dataset={"times": [Delta(dtype="float64")]},
        chunk_duration=f"{time_chunk_s}s",
        n_jobs=n_jobs,
        progress_bar=True,
    )


def open_lfps_dataarray(
    lfp_zarr: str | pathlib.Path, *, chunks: str | dict = "auto"
) -> xr.DataArray:
    """Open an LFP Zarr store as a lazy (dask-backed) xarray DataArray.

    dims ``(time, channel)``; coords ``time`` (session-relative seconds),
    ``channel`` (names), ``tetrode`` (``TTk`` labels, if present), and ``x``/``y``
    (probe locations). ``attrs={"fs", "units"}``.
    """
    lfp_zarr = pathlib.Path(lfp_zarr)
    rec = si.load(str(lfp_zarr))
    fs = rec.get_sampling_frequency()
    times = np.asarray(rec.get_times(segment_index=0))

    prop_keys = rec.get_property_keys()
    name_key = "channel_name" if "channel_name" in prop_keys else "channel_names"
    channel_names = (
        rec.get_property(name_key) if name_key in prop_keys else rec.get_channel_ids()
    )

    coords: dict = {
        "time": ("time", times),
        "channel": ("channel", np.asarray(channel_names)),
    }
    if "tetrode" in prop_keys:
        coords["tetrode"] = ("channel", np.asarray(rec.get_property("tetrode")))
    try:
        loc = rec.get_channel_locations()
        coords["x"] = ("channel", loc[:, 0])
        coords["y"] = ("channel", loc[:, 1])
    except Exception:
        pass

    zg = zarr.open(str(lfp_zarr), mode="r")
    traces = da.from_zarr(zg["traces_seg0"])
    if isinstance(chunks, dict) or chunks == "auto":
        traces = traces.rechunk(chunks if isinstance(chunks, dict) else "auto")

    return xr.DataArray(
        traces,
        dims=("time", "channel"),
        coords=coords,
        attrs={"fs": fs, "units": "microvolts"},
    )

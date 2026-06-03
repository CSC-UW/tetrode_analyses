"""Instantaneous and STFT bandpower extraction for tetrode LFPs.

Two complementary band-power estimators, both averaged **per tetrode** (mean over
each tetrode's 4 channels) so on-disk products are ~4x smaller than per-channel:

- :func:`extract_instantaneous_bandpower` — bandpass + Hilbert envelope, following
  ``wisc_ecephys_tools.rats.pipeline.get_instantaneous_power.do_all_delta`` but
  with **no bipolar referencing** (tetrodes are referenced at acquisition).
- :func:`compute_stft_bandpowers` — short-time Fourier transform PSDs summed within
  bands, following ``findlay2025a.pipeline.compute_cx_bandpowers_and_psds`` (same
  4 s STFT chunk, DPSS window, q=2 decimation, and band definitions).

Inputs are LFP DataArrays from :func:`tetrode_analyses.lfp.open_lfps_dataarray`:
dims ``(time, channel)``, ``attrs["fs"]``, and a ``tetrode`` coord (``"TTk"``)
along ``channel``.
"""

from __future__ import annotations

import pathlib
import re

import dask.array as dsa
import numpy as np
import pandas as pd
import xarray as xr
from ecephys.npsig import get_n_fft

# Heavy signal-processing functions must be imported from xrsig.core directly;
# ecephys.xrsig.__init__ deliberately does not re-export them (slow imports).
from ecephys.xrsig.core import (
    antialiasing_filter,
    butter_bandpass,
    decimate_timeseries,
    hilbert,
    stft_psd,
)

#: Frequency bands (Hz), matching ``findlay2025a.constants.Bands``.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4),
    "vlad": (2, 6),  # a.k.a. eta
    "theta": (5, 10),
    "sigma": (9, 16),
    "gamma": (30, 150),
}

#: Default STFT chunk duration (seconds), matching findlay2025a.
STFT_CHUNK_DURATION: float = 4.0
#: Default STFT decimation factor, matching findlay2025a cortical pipeline.
STFT_DECIMATE_Q: int = 2
#: Default instantaneous-power decimation factor: 625 Hz / 5 = 125 Hz, matching the
#: ~125 Hz target of the rat ``do_all_delta`` (qs=[10, 2] on 2500 Hz LFP).
INST_DECIMATE_Q: int = 5


def _tetrode_int(label: object) -> int:
    """Integer index parsed from a ``"TTk"`` tetrode label (for natural sort)."""
    return int(re.sub(r"\D", "", str(label)))


def average_within_tetrode(da: xr.DataArray) -> xr.DataArray:
    """Mean over each tetrode's channels; returns a ``tetrode``-indexed DataArray.

    Reduces the ``channel`` dimension via the ``tetrode`` coord, in natural
    (``TT1, TT2, ..., TT10``) order. Done by explicit selection rather than
    ``groupby`` so it stays lazy on dask-backed (chunked) arrays.
    """
    labels = np.asarray(da["tetrode"].values)
    tetrodes = sorted(set(labels.tolist()), key=_tetrode_int)
    means = [
        da.isel(channel=np.flatnonzero(labels == tt)).mean(dim="channel")
        for tt in tetrodes
    ]
    # Use a fixed-width unicode array (not a pandas object Index) for the new
    # ``tetrode`` coord so it serializes to zarr v2 without an object codec.
    dim = xr.DataArray(np.array(tetrodes), dims="tetrode", name="tetrode")
    out = xr.concat(means, dim=dim)
    out.attrs = dict(da.attrs)
    return out


def extract_instantaneous_bandpower(
    lfp: xr.DataArray,
    lowcut: float,
    highcut: float,
    *,
    order: int = 2,
    decimate_q: int = INST_DECIMATE_Q,
) -> xr.DataArray:
    """Per-tetrode instantaneous band power via bandpass + Hilbert envelope.

    Decimates the LFP, bandpasses (zero-phase Butterworth), takes the analytic
    signal, squares its magnitude, then averages within tetrode. No bipolar
    referencing. Returns dims ``(tetrode, time)``, name ``"pwr"``, with the
    decimated sampling rate in ``attrs["fs"]``.
    """
    lfp = decimate_timeseries(lfp, decimate_q)
    # The dask bandpass/Hilbert use map_overlap and require each time chunk to be
    # longer than the filter impulse response; rechunk to generous ~300 s chunks
    # (and a single channel chunk) so chunking from the source store can't underflow.
    lfp = lfp.chunk({"time": int(lfp.fs * 300), "channel": -1})
    filt = butter_bandpass(lfp, lowcut, highcut, order)
    analytic = hilbert(filt)
    power = analytic.copy()
    power.data = dsa.square(dsa.abs(analytic.data))
    power.attrs = dict(lfp.attrs)
    ipwr = average_within_tetrode(power)
    ipwr.attrs.update(
        units="microvolts^2", lowcut=lowcut, highcut=highcut, filter_order=order
    )
    return ipwr.rename("pwr")


def compute_stft_bandpowers(
    lfp: xr.DataArray,
    *,
    bands: dict[str, tuple[float, float]] = BANDS,
    stft_chunk_s: float = STFT_CHUNK_DURATION,
    decimate_q: int = STFT_DECIMATE_Q,
) -> xr.Dataset:
    """Per-tetrode band-power timeseries from STFT PSDs.

    Decimates the LFP (anti-aliased stride), computes per-channel STFT PSDs
    (DPSS window, ``stft_chunk_s`` chunks), averages PSDs within tetrode, then
    sums power within each band. Returns a Dataset with one ``(tetrode, time)``
    variable per band. Discontinuous segments (wall-clock gaps between Open Ephys
    experiments) are handled by ``xrsig.stft_psd``.
    """
    lfp = antialiasing_filter(lfp, q=decimate_q)
    lfp = lfp.isel(time=slice(None, None, decimate_q))
    lfp.attrs["fs"] = lfp.fs / decimate_q

    _, (n_fft, _) = get_n_fft(lfp.fs, s=stft_chunk_s)
    spgs = stft_psd(lfp, n_fft=n_fft)  # (channel, frequency, time)
    spgs = average_within_tetrode(spgs)  # (tetrode, frequency, time)

    ds = xr.Dataset(
        {
            name: spgs.sel(frequency=slice(lo, hi)).sum(dim="frequency")
            for name, (lo, hi) in bands.items()
        }
    )
    ds.attrs.update(
        fs=float(lfp.fs),
        stft_n_fft=int(n_fft),
        stft_chunk_s=float(stft_chunk_s),
        decimate_q=int(decimate_q),
        units="microvolts^2",
    )
    return ds


def save_instantaneous_power(
    da: xr.DataArray, path: str | pathlib.Path
) -> pathlib.Path:
    """Write per-tetrode instantaneous power to a Zarr store (overwriting)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # zarr V3's StoreLike type excludes str/Path, though xarray accepts them.
    da.to_dataset(name="pwr").to_zarr(path, mode="w")  # ty: ignore[invalid-argument-type]
    return path


def open_instantaneous_power(path: str | pathlib.Path) -> xr.DataArray:
    """Open a per-tetrode instantaneous-power Zarr store as a DataArray."""
    return xr.open_zarr(str(path))["pwr"]


def save_stft_bandpowers(ds: xr.Dataset, path: str | pathlib.Path) -> pathlib.Path:
    """Write per-tetrode STFT band powers to a Zarr store (overwriting).

    Zarr (rather than netCDF) for consistency with the other tetrode products and
    to avoid the HDF engine choking on the unicode ``tetrode`` coordinate.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # zarr V3's StoreLike type excludes str/Path, though xarray accepts them.
    ds.to_zarr(path, mode="w")  # ty: ignore[invalid-argument-type]
    return path


def open_stft_bandpowers(path: str | pathlib.Path) -> xr.Dataset:
    """Open a per-tetrode STFT band-power Zarr store."""
    return xr.open_zarr(str(path))


# --- Outlier replacement (kernel-density-gap method) -------------------------
# Ported from findlay2025a.pipeline.extract_cx_ipower.replace_outliers_kd: find
# the first gap (run of empty histogram bins) in the value distribution and treat
# everything above it as an artifact. Usually more conservative (higher threshold)
# than even np.nanquantile(x, 0.9999). Applied at plot time here, not before save.


def _find_first_zero_run(a: np.ndarray, n: int = 1) -> int | None:
    """Index of the first run of ``n`` consecutive zeros in ``a`` (``None`` if none)."""
    for i in range(len(a) - n + 1):
        if not any(a[i : i + n]):
            return i
    return None


def _get_threshold_table(hist: np.ndarray, bin_edges: np.ndarray) -> pd.DataFrame:
    """Map each empty-bin run length to the value where the first such gap opens."""
    zero_runs = np.split(np.arange(len(hist)), np.where(np.diff(hist == 0))[0] + 1)
    zero_runs = [run for run in zero_runs if hist[run[0]] == 0]
    run_lengths = [len(run) for run in zero_runs]
    thresholds = {}
    for n in range(1, max(run_lengths, default=0) + 1):
        ix = _find_first_zero_run(hist, n)
        if ix is not None:
            thresholds[n] = bin_edges[ix]
    df = pd.DataFrame(
        {"run_length": list(thresholds.keys()), "threshold": list(thresholds.values())}
    )
    # Collapse to distinct thresholds, keeping the smallest run length for each.
    return df.groupby("threshold").first().reset_index()


def replace_outliers(
    x: np.ndarray,
    *,
    threshold_ix: int = 0,
    bins: int = 1000,
    fill_value: float = np.nan,
) -> tuple[float, np.ndarray]:
    """Replace values above the first histogram-gap threshold with ``fill_value``.

    Returns ``(threshold, cleaned)``. ``threshold_ix`` selects which gap to use
    (0 = first/most aggressive). If the distribution has no empty-bin gap, nothing
    is replaced. Operates on a copy; ``x`` must be float to hold NaN fills.
    """
    x = np.asarray(x, dtype=float).copy()
    finite = x[~np.isnan(x)]
    if finite.size == 0:
        return np.inf, x
    hist, bin_edges = np.histogram(finite, bins=bins)
    tdf = _get_threshold_table(hist, bin_edges)
    if tdf.empty:
        return np.inf, x
    threshold = float(tdf.loc[threshold_ix, "threshold"])
    x[x > threshold] = fill_value
    return threshold, x


def replace_outliers_per_tetrode(
    da: xr.DataArray, **kwargs
) -> xr.DataArray:
    """Apply :func:`replace_outliers` independently to each tetrode's timeseries.

    ``da`` has dims ``(tetrode, time)``; each tetrode gets its own threshold (its
    artifact distribution differs). Returns a new DataArray with outliers as NaN.
    """
    out = da.astype(float).copy()
    values = out.values
    for i in range(values.shape[0]):
        _, values[i] = replace_outliers(values[i], **kwargs)
    return out

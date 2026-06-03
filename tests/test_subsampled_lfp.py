"""Tests for the subsampled-LFP step: lead-channel selection, integer decimation,
anti-aliasing, and channel-list override.

These build a tiny synthetic LFP Zarr store, so no NFS/production data is needed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tetrode_analyses.lfp import (
    make_subsampled_lfp,
    open_lfps_dataarray,
    tetrode_lead_channel_ids,
)


def _toy_lfp_recording(
    tmp_path: pathlib.Path,
    *,
    fs: float = 600.0,
    n_tetrodes: int = 3,
    dur_s: float = 20.0,
    f_low: float = 5.0,
    f_high: float = 120.0,
):
    """A grouped synthetic LFP recording = low-freq tone + a tone above the
    post-decimation Nyquist (to probe anti-aliasing), with a session-relative
    time vector. ``fs=600`` so ``/6`` gives a clean 100 Hz (Nyquist 50)."""
    si = pytest.importorskip("spikeinterface.core")
    n_ch = 4 * n_tetrodes
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    signal = (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t)).astype("float32")
    traces = np.tile(signal[:, None], (1, n_ch))
    ids = [f"CH{i}" for i in range(n_ch)]
    rec = si.NumpyRecording([traces], sampling_frequency=fs, channel_ids=ids)
    rec.set_property("group", np.repeat(np.arange(n_tetrodes), 4))
    rec.set_property(
        "tetrode", np.repeat([f"TT{k + 1}" for k in range(n_tetrodes)], 4)
    )
    rec.set_times(t + 100.0, segment_index=0)  # mimic real session-relative clock
    return rec


def _toy_lfp_zarr(tmp_path: pathlib.Path, **kwargs):
    rec = _toy_lfp_recording(tmp_path, **kwargs)
    zpath = tmp_path / "lfp.zarr"
    rec.save(format="zarr", folder=str(zpath))
    return zpath


def test_tetrode_lead_channel_ids_from_group(tmp_path: pathlib.Path) -> None:
    rec = _toy_lfp_recording(tmp_path, n_tetrodes=3)
    leads = tetrode_lead_channel_ids(rec)
    assert list(map(str, leads)) == ["CH0", "CH4", "CH8"]  # first of each tetrode


def test_tetrode_lead_channel_ids_requires_grouping(tmp_path: pathlib.Path) -> None:
    si = pytest.importorskip("spikeinterface.core")
    rec = si.NumpyRecording(
        [np.zeros((10, 4), dtype="float32")],
        sampling_frequency=600.0,
        channel_ids=[f"CH{i}" for i in range(4)],
    )
    with pytest.raises(ValueError, match="group.*tetrode"):
        tetrode_lead_channel_ids(rec)


def test_make_subsampled_lfp_decimates_and_antialiases(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("spikeinterface.preprocessing")
    # fs=600, resample to 100 (÷6, integer ratio → scipy.signal.decimate path).
    zpath = _toy_lfp_zarr(tmp_path, fs=600.0, n_tetrodes=3, f_low=5.0, f_high=120.0)
    out = tmp_path / "sub.zarr"

    make_subsampled_lfp(zpath, out, resample_rate=100, n_jobs=1)

    da = open_lfps_dataarray(out)
    # ÷6 from 600 Hz, one lead per tetrode, tetrode coord carried through.
    assert float(da.attrs["fs"]) == pytest.approx(100.0)
    assert da.sizes["channel"] == 3
    assert "tetrode" in da.coords
    # session-relative time offset (set_times +100 s) survives, decimated.
    assert float(da["time"].values[0]) == pytest.approx(100.0, abs=0.05)

    # Anti-aliasing: the 120 Hz input tone would alias to |120-100|=20 Hz after
    # naive decimation; resample's scipy.signal.decimate must suppress it while
    # the 5 Hz tone survives.
    x = np.asarray(da.isel(channel=0).values)
    x = x[30:-30]  # trim filter edge transients
    amp = (2.0 / x.size) * np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / 100.0)
    amp_5 = amp[np.argmin(np.abs(freqs - 5.0))]
    amp_20 = amp[np.argmin(np.abs(freqs - 20.0))]
    assert amp_5 > 0.5  # passband tone (input amplitude 1.0) preserved
    assert amp_20 < 0.05  # aliased tone strongly attenuated
    assert amp_5 > 20 * amp_20


def test_make_subsampled_lfp_explicit_channels(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("spikeinterface.preprocessing")
    zpath = _toy_lfp_zarr(tmp_path, fs=600.0, n_tetrodes=3)
    out = tmp_path / "sub_explicit.zarr"

    make_subsampled_lfp(zpath, out, channel_ids=["CH1", "CH5"], resample_rate=100, n_jobs=1)

    da = open_lfps_dataarray(out)
    assert da.sizes["channel"] == 2
    assert set(map(str, da["channel"].values)) == {"CH1", "CH5"}


def test_make_subsampled_lfp_rejects_indivisible_rate(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("spikeinterface.preprocessing")
    zpath = _toy_lfp_zarr(tmp_path, fs=600.0)
    out = tmp_path / "sub_bad.zarr"
    # 600 is not divisible by 104 → would force resample's FFT fallback; rejected.
    with pytest.raises(ValueError, match="evenly divide"):
        make_subsampled_lfp(zpath, out, resample_rate=104)


# --- End-to-end against the production sub-LFP store (requires NFS) ------------

SUB_LFP = pathlib.Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/2026-05-27_09-07-52.lfp.125hz.zarr"
)


@pytest.mark.requires_nfs
def test_production_sub_lfp_shape() -> None:
    if not SUB_LFP.exists():
        pytest.skip(f"sub-LFP store not built: {SUB_LFP}")
    da = open_lfps_dataarray(SUB_LFP)
    assert da.sizes["channel"] == 16  # one lead per tetrode
    assert float(da.attrs["fs"]) == pytest.approx(125.0)
    assert "tetrode" in da.coords

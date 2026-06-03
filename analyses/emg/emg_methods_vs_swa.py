"""Generate synthetic EMG (both methods) for TTM-001 / TTM-NOD and plot it
alongside SWA (delta power) with a lights-on/off + sleep-deprivation overlay.

Two EMG estimators are computed on the same decimated LFP (see
`ecephys.emg_from_lfp`):
  - "per_window": exact per-window mean pairwise Pearson correlation (default).
  - "global":     faster amplitude-weighted global-normalization approximation.

SWA is the pre-computed instantaneous delta (0.5-4 Hz) power artifact, averaged
across tetrodes. Lights/deprivation come from experiment_params.json.

Run (validate on a short window first, then the full ~48 h session):
    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/emg/emg_methods_vs_swa.py
    uv run python ../tetrode_analyses/analyses/emg/emg_methods_vs_swa.py --full
"""

from __future__ import annotations

import argparse
import pathlib
import time

import dask.array as da
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spikeinterface as si  # noqa: E402
import xarray as xr  # noqa: E402
import zarr  # noqa: E402

from ecephys.xrsig import core as xrsig  # noqa: E402
from tetrode_analyses import experiment as exp  # noqa: E402
from tetrode_analyses import power  # noqa: E402

SUBJECT, EXPERIMENT = "TTM-001", "TTM-NOD"
FS_IN, DECIM_Q = 30000, 20  # -> 1500 Hz, max decimation synthetic_emg allows
TETRODES = ("TT1", "TT8", "TT16")  # one channel each; max spatial separation
EMG_TARGET_SF = 20  # synthetic_emg default output rate

OUTDIR = exp.experiment_dir(SUBJECT, EXPERIMENT) / "figures"
EMG_ZARR = exp.experiment_dir(SUBJECT, EXPERIMENT) / "synthetic_emg_methods.zarr"
LIGHT_COLORS = {"on": "gold", "off": "0.35"}


# --------------------------------------------------------------------------
# EMG generation
# --------------------------------------------------------------------------
def load_segment_uv(zg, gains, col_idx, start_sample, end_sample, time_slice=None):
    """Lazy (dask) uV DataArray of the selected channels for a sample range."""
    traces = da.from_zarr(zg["traces_seg0"])  # (n_samples, 64) int16
    seg = traces[start_sample:end_sample][:, col_idx]
    if time_slice is not None:
        seg = seg[time_slice]
    seg_uv = seg.astype("float32") * gains[col_idx].astype("float32")
    return xr.DataArray(
        seg_uv, dims=("time", "channel"), attrs={"fs": FS_IN, "units": "microvolts"}
    )


def segment_emgs(da3, t_start, t_end):
    """Decimate to 1500 Hz once, then compute both EMG methods (single filter).

    Returns an xr.Dataset with `per_window` and `global` (time,) variables.
    """
    t0 = time.perf_counter()
    dec = xrsig.decimate_timeseries(da3, q=DECIM_Q).load()
    dec = dec.assign_coords(time=np.linspace(t_start, t_end, dec.sizes["time"]))
    print(f"  decimated -> {dec.attrs['fs']:.0f} Hz, {dec.sizes['time']} samples "
          f"({dec.sizes['time'] / dec.attrs['fs'] / 3600:.2f} h) in "
          f"{time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    ds = xrsig.synthetic_emg(dec)  # method="both" (default): one filter, both methods
    print(f"  emg (both) -> {ds.sizes['time']} samples in {time.perf_counter() - t0:.1f}s "
          f"(per_window {float(ds['per_window'].min()):.3f}..{float(ds['per_window'].max()):.3f}, "
          f"global {float(ds['global'].min()):.3f}..{float(ds['global'].max()):.3f})")
    return ds


def generate_emg(full: bool):
    """Return an xr.Dataset with one (time,) EMG variable per method."""
    params = exp.load_experiment_params(exp.experiment_params_path(SUBJECT, EXPERIMENT))
    root = pathlib.Path(params.openephys_session)
    src = root / f"{root.name}.blosc-zstd.zarr"
    rec = si.read_zarr(str(src))
    tetrode = np.asarray(rec.get_property("tetrode"))
    gains = np.asarray(rec.get_property("gain_to_uV"))
    col_idx = np.array([int(np.flatnonzero(tetrode == tt)[0]) for tt in TETRODES])
    print(f"channels (tetrodes {TETRODES}): cols {col_idx.tolist()}")
    st = pd.read_csv(root / "slice_table.csv")
    zg = zarr.open(str(src), mode="r")

    if full:
        print("FULL run: both experiments, processed independently then concatenated")
        segs = []
        for _, row in st.iterrows():
            print(f"segment {row.experiment_name}: samples "
                  f"{int(row.start_sample)}:{int(row.end_sample)}  "
                  f"t {row.t_start:.0f}..{row.t_end:.0f}s")
            da3 = load_segment_uv(zg, gains, col_idx, int(row.start_sample),
                                  int(row.end_sample))
            segs.append(segment_emgs(da3, row.t_start, row.t_end))
        ds = xr.concat(segs, dim="time")
    else:
        # 30-min validation window, 10 h into experiment1
        row = st.iloc[0]
        dur_s, start_s = 1800, 10 * 3600
        s0 = int(row.start_sample) + start_s * FS_IN
        s1 = s0 + dur_s * FS_IN
        print(f"WINDOW run: experiment1, {dur_s / 60:.0f} min from {start_s / 3600:.0f} h")
        da3 = load_segment_uv(zg, gains, col_idx, int(row.start_sample),
                              int(row.end_sample),
                              time_slice=slice(s0 - int(row.start_sample),
                                               s1 - int(row.start_sample)))
        ds = segment_emgs(da3, row.t_start + start_s, row.t_start + start_s + dur_s)

    return ds, params


# --------------------------------------------------------------------------
# SWA
# --------------------------------------------------------------------------
def load_swa():
    """Mean instantaneous delta (0.5-4 Hz) power across tetrodes, session seconds."""
    path = exp.experiment_dir(SUBJECT, EXPERIMENT) / "delta.idelta.zarr"
    idelta = power.open_instantaneous_power(path)
    idelta = power.replace_outliers_per_tetrode(idelta.compute())
    return idelta.mean("tetrode")  # (time,)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _overlay(ax, params, hours=True):
    div = 3600.0 if hours else 1.0
    x0, x1 = ax.get_xlim()
    intervals, labels = exp.get_light_dark_periods(params)
    for (a, b), lab in zip(intervals, labels):
        a, b = max(a / div, x0), min(b / div, x1)
        if b > a:
            ax.axvspan(a, b, ymin=0.0, ymax=0.04, color=LIGHT_COLORS[lab],
                       zorder=1000, ec="none", clip_on=False)
    dep = exp.get_deprivation_period(params)
    if dep is not None:
        ax.axvspan(dep[0] / div, dep[1] / div, color="red", alpha=0.10, zorder=0)
    ax.set_xlim(x0, x1)


def _trace(ax, da_, smoothing, color, label, ymax_pct=None, xlim_h=None):
    d = da_
    if smoothing > 1:
        d = d.rolling(time=smoothing, center=True, min_periods=1).mean()
    x = d["time"].values / 3600.0
    y = d.values
    ax.plot(x, y, lw=0.6, color=color)
    ax.set_ylabel(label)
    ax.margins(x=0)
    # Clip the y-axis to a percentile of the *visible* data so a transient
    # artifact doesn't flatten the trace (display only).
    vis = (x >= xlim_h[0]) & (x <= xlim_h[1]) if xlim_h is not None else np.isfinite(y)
    yv = y[vis & np.isfinite(y)]
    if ymax_pct is not None and yv.size:
        top = np.nanpercentile(yv, ymax_pct)
        if np.isfinite(top) and top > 0:
            ax.set_ylim(min(0.0, float(yv.min())), top * 1.08)


def plot_panels(ds, swa, params, out_png, *, xlim_h=None, emg_smooth=1, swa_smooth=1,
                mode="full", title=""):
    # Clip transient artifacts so traces stay legible. per_window is bounded to
    # ~[0,1]; global and SWA can spike, so clip them to a high percentile of the
    # visible window.
    g_pct = 99.8 if mode == "full" else 99.0
    swa_pct = 99.5 if mode == "full" else 98.0
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _trace(axs[0], ds["per_window"], emg_smooth, "C0", "EMG\nper_window", xlim_h=xlim_h)
    _trace(axs[1], ds["global"], emg_smooth, "C1", "EMG\nglobal", ymax_pct=g_pct,
           xlim_h=xlim_h)
    _trace(axs[2], swa, swa_smooth, "C2", "SWA\n(delta µV²)", ymax_pct=swa_pct,
           xlim_h=xlim_h)
    axs[0].axhline(1.0, ls=":", c="gray", lw=0.7)  # valid-correlation ceiling
    if xlim_h is not None:
        for ax in axs:
            ax.set_xlim(*xlim_h)
    for ax in axs:
        _overlay(ax, params)
    axs[-1].set_xlabel("session time (h)   [gold=lights-on, dark=lights-off, red=deprivation]")
    axs[0].set_title(title)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def _extreme_time(da_, fs, kind, exclude=None, smooth_s=30):
    """Session-time (s) of the max/min smoothed value, excluding an interval.

    Selection uses heavy smoothing so transient artifacts don't dominate.
    """
    sm = da_.rolling(time=int(fs * smooth_s), center=True, min_periods=1).mean()
    t = sm["time"].values
    y = sm.values.astype(float).copy()
    if exclude is not None:
        y[(t >= exclude[0]) & (t <= exclude[1])] = np.inf if kind == "min" else -np.inf
    return float(t[np.nanargmin(y) if kind == "min" else np.nanargmax(y)])


def zoom_windows(ds, swa, params):
    """Data-driven (label, center_session_seconds) for minute-scale zooms.

    Selects epochs by EMG (a clean state indicator robust to SWA artifacts):
    the lowest-EMG minute (NREM-like, expect high SWA), the highest-EMG minute
    outside deprivation (active wake, expect low SWA), and the deprivation
    midpoint.
    """
    dep = exp.get_deprivation_period(params)
    pw, fs = ds["per_window"], EMG_TARGET_SF
    wins = [
        ("NREM_lowEMG", _extreme_time(pw, fs, "min", exclude=dep)),
        ("wake_highEMG", _extreme_time(pw, fs, "max", exclude=dep)),
    ]
    if dep is not None:
        wins.append(("deprivation", (dep[0] + dep[1]) / 2))
    return wins


# --------------------------------------------------------------------------
def main(full: bool, plot_only: bool = False):
    if plot_only:
        full = True
        params = exp.load_experiment_params(
            exp.experiment_params_path(SUBJECT, EXPERIMENT)
        )
        ds = xr.open_zarr(EMG_ZARR)
        print(f"loaded {EMG_ZARR} ({ds.sizes['time']} samples)")
    else:
        ds, params = generate_emg(full)
        if full:
            ds.to_zarr(EMG_ZARR, mode="w")
            print(f"saved {EMG_ZARR}")
    swa = load_swa()
    print(f"SWA: {swa.sizes['time']} samples @ {swa.attrs.get('fs', '?')} Hz, "
          f"time {float(swa['time'].min()):.0f}..{float(swa['time'].max()):.0f}s")

    tag = "full" if full else "window"
    plot_panels(ds, swa, params, OUTDIR / f"emg_methods_vs_swa_{tag}.png",
                emg_smooth=(101 if full else 11), swa_smooth=(1250 if full else 250),
                mode=("full" if full else "window"),
                title=f"{SUBJECT} {EXPERIMENT}: EMG (per_window vs global) & SWA "
                      f"({'full ~48 h' if full else '30 min window'})")

    if full:
        for label, center_s in zoom_windows(ds, swa, params):
            c_h = center_s / 3600.0
            half_h = 5.0 / 60.0  # 10-min window
            plot_panels(ds, swa, params,
                        OUTDIR / f"emg_methods_vs_swa_zoom_{label}.png",
                        xlim_h=(c_h - half_h, c_h + half_h),
                        emg_smooth=1, swa_smooth=1250, mode="zoom",
                        title=f"{SUBJECT} {EXPERIMENT}: 10-min zoom @ {label} "
                              f"(session {center_s / 3600:.2f} h)")
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="process full ~48 h session")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip generation; load saved EMG zarr and (re)plot")
    args = ap.parse_args()
    main(args.full, plot_only=args.plot_only)

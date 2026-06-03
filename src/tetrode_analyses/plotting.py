"""SWA (delta power) timetrace plotting with light/dark and deprivation overlays.

Mirrors ``findlay2025a.plotting`` (timetrace) and
``wisc_ecephys_tools.rats.cnd_hgs`` (lights overlay) so tetrode SWA figures match
the rest of the workspace, but operates on session-relative seconds and the
per-tetrode power products from :mod:`tetrode_analyses.power`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from tetrode_analyses import experiment as exp

LIGHT_COLORS = {"on": "gold", "off": "gray"}


def plot_timetrace(
    da: xr.DataArray,
    ax: plt.Axes,
    smoothing: int = 0,
    q: int = 1,
    color: str = "black",
    linewidth: float = 0.5,
) -> plt.Axes:
    """Plot a 1-D timeseries: optional rolling-mean smoothing then decimation.

    ``smoothing`` is a rolling-window length in samples; ``q`` decimates by taking
    every ``q``-th sample (purely for plot density). Mirrors
    ``findlay2025a.plotting.plot_timetrace``.
    """
    if smoothing:
        # min_periods=1 so NaN outliers (from replace_outliers) are skipped rather
        # than blanking the whole smoothing window.
        da = da.rolling(time=smoothing, center=True, min_periods=1).mean()
    if q > 1:
        da = da.isel(time=slice(None, None, q))
    sns.lineplot(
        x=da["time"].values, y=da.values, color=color, linewidth=linewidth, ax=ax
    )
    ax.set(xlabel=None, xmargin=0)
    return ax


def plot_swa_timetrace(da: xr.DataArray, ax: plt.Axes, **kwargs) -> plt.Axes:
    """Plot a slow-wave-activity (delta power) timetrace."""
    plot_timetrace(da, ax, **kwargs)
    ax.set(ylabel="SWA")
    return ax


def plot_lights_overlay(
    intervals: list[tuple[float, float]],
    labels: list[str],
    ax: plt.Axes,
    ymin: float = 0.0,
    ymax: float = 0.03,
    colors: dict[str, str] = LIGHT_COLORS,
    alpha: float = 1.0,
    zorder: int = 1000,
) -> plt.Axes:
    """Shade light/dark periods as spans along the x-axis (session seconds).

    ``ymin``/``ymax`` are axes-fraction bounds (a thin band at the bottom by
    default). Spans are clipped to the current x-limits. Mirrors
    ``wisc_ecephys_tools.rats.cnd_hgs.plot_lights_overlay``.
    """
    xlim = ax.get_xlim()
    for (t_on, t_off), label in zip(intervals, labels):
        t_on = max(t_on, xlim[0])
        t_off = min(t_off, xlim[1])
        if t_off <= t_on:
            continue
        ax.axvspan(
            t_on,
            t_off,
            ymin=ymin,
            ymax=ymax,
            color=colors[label],
            alpha=alpha,
            zorder=zorder,
            ec="none",
            clip_on=False,
        )
    ax.set_xlim(xlim)
    return ax


def plot_deprivation_overlay(
    period: tuple[float, float],
    ax: plt.Axes,
    ymin: float = 0.0,
    ymax: float = 1.0,
    color: str = "red",
    alpha: float = 0.12,
    zorder: int = 1,
) -> plt.Axes:
    """Shade the sleep-deprivation window (session seconds) across the axis."""
    t_start, t_end = period
    ax.axvspan(
        t_start, t_end, ymin=ymin, ymax=ymax, color=color, alpha=alpha, zorder=zorder
    )
    return ax


def _robust_ymax(ax: plt.Axes, percentile: float) -> None:
    """Clip the upper y-limit to a percentile of the plotted line(s) (display only).

    Zooms past acquisition transients without altering the underlying data.
    """
    y = np.concatenate([line.get_ydata() for line in ax.lines]) if ax.lines else None
    if y is None or not np.isfinite(y).any():
        return
    top = np.nanpercentile(y, percentile)
    if np.isfinite(top) and top > 0:
        ax.set_ylim(0, top)


def plot_swa_overview(
    da: xr.DataArray,
    params: exp.ExperimentParams,
    ax: plt.Axes,
    *,
    smoothing: int = 0,
    q: int = 1,
    title: str | None = None,
    ymax_percentile: float | None = None,
) -> plt.Axes:
    """Plot one SWA trace spanning the recording with lights + deprivation overlays.

    ``da`` is a 1-D ``(time,)`` SWA timeseries (e.g. mean across tetrodes).
    ``ymax_percentile`` (e.g. 99.5) clips the upper y-limit for display so
    transient artifacts don't compress the trace; ``None`` leaves it autoscaled.
    """
    ax.set_xlim(float(da["time"].min()), float(da["time"].max()))
    deprivation = exp.get_deprivation_period(params)
    if deprivation is not None:
        plot_deprivation_overlay(deprivation, ax)
    plot_swa_timetrace(da, ax, smoothing=smoothing, q=q)
    if ymax_percentile is not None:
        _robust_ymax(ax, ymax_percentile)
    intervals, labels = exp.get_light_dark_periods(params)
    plot_lights_overlay(intervals, labels, ax)
    ax.set_xlabel("Session time (s)")
    if title:
        ax.set_title(title)
    return ax


def plot_swa_small_multiples(
    da: xr.DataArray,
    params: exp.ExperimentParams,
    *,
    smoothing: int = 0,
    q: int = 1,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ymax_percentile: float | None = None,
) -> plt.Figure:
    """Stacked per-tetrode SWA traces sharing one lights/deprivation overlay.

    ``da`` has dims ``(tetrode, time)``. ``ymax_percentile`` clips each row's
    upper y-limit for display (see :func:`plot_swa_overview`). Returns the Figure.
    """
    tetrodes = list(da["tetrode"].values)
    n = len(tetrodes)
    if figsize is None:
        figsize = (14, max(6, 0.7 * n))
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    intervals, labels = exp.get_light_dark_periods(params)
    deprivation = exp.get_deprivation_period(params)
    xlim = (float(da["time"].min()), float(da["time"].max()))
    for ax, tt in zip(axes, tetrodes):
        ax.set_xlim(*xlim)
        if deprivation is not None:
            plot_deprivation_overlay(deprivation, ax)
        plot_timetrace(da.sel(tetrode=tt), ax, smoothing=smoothing, q=q)
        if ymax_percentile is not None:
            _robust_ymax(ax, ymax_percentile)
        plot_lights_overlay(intervals, labels, ax)
        ax.set(ylabel=str(tt), yticks=[])
    axes[-1].set_xlabel("Session time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

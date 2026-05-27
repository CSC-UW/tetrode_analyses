"""Loading helpers for Open Ephys tetrode acquisitions."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from open_ephys import analysis as oea


Layout = Literal["2d_flat_index", "2d_multi_index", "3d"]


def _load_channel_map(settings_path: pathlib.Path) -> dict:
    """Parse a record node's ``settings.xml`` Channel Map processor.

    Returns a dict with the channel order found in the Channel Map, the
    per-channel enabled flags, and the tetrode → channel groupings derived
    from contiguous groups of four. Tetrode numbers are 1-indexed in both
    ``tt2ch_ixs`` and ``tt2ch_names`` so callers can refer to physical
    tetrodes by their natural labels.
    """
    tree = ET.parse(settings_path)
    root = tree.getroot()

    channel_map = next(
        p
        for p in root.findall("./SIGNALCHAIN/PROCESSOR")
        if p.get("name") == "Channel Map"
    )
    stream = channel_map.find("./CUSTOM_PARAMETERS/STREAM")

    oe_indices = [int(ch.get("index")) for ch in stream.findall("CH")]
    enabled = [int(ch.get("enabled")) for ch in stream.findall("CH")]

    n_tetrodes = len(oe_indices) // 4
    tt2ch_ixs = {
        tt: oe_indices[(tt - 1) * 4 : tt * 4] for tt in range(1, n_tetrodes + 1)
    }
    tt2ch_names = {
        tt: [(tt - 1) * 4 + i for i in range(1, 5)]
        for tt in range(1, n_tetrodes + 1)
    }

    return {
        "oe_indices": oe_indices,
        "enabled": enabled,
        "tt2ch_ixs": tt2ch_ixs,
        "tt2ch_names": tt2ch_names,
    }


def _get_recording(node, experiment_index: int, recording_index: int):
    for rec in node.recordings:
        if (
            rec.experiment_index == experiment_index
            and rec.recording_index == recording_index
        ):
            return rec
    available = [(r.experiment_index, r.recording_index) for r in node.recordings]
    raise ValueError(
        f"No recording with experiment_index={experiment_index}, "
        f"recording_index={recording_index} on record node. "
        f"Available (experiment, recording) pairs: {available}"
    )


def print_session_tree(acq_dir: str | pathlib.Path) -> None:
    """Print a compact tree of the Open Ephys session at ``acq_dir``.

    Lists each record node, its experiments and recordings, and every
    continuous stream within each recording — annotated with channel
    count, sample count, sampling rate, and total duration in seconds.
    Stream indices match what :func:`load_tetrode_dataarray` expects.
    """
    acq_dir = pathlib.Path(acq_dir)
    session = oea.Session(str(acq_dir))

    print(f"Session: {acq_dir}")
    for node_ix, node in enumerate(session.recordnodes):
        node_name = pathlib.Path(node.directory).name
        fmt = getattr(node, "format", "unknown")
        print(f"  {node_name} ({fmt})  [record_node_index={node_ix}]")

        by_experiment: dict[int, list] = defaultdict(list)
        for rec in node.recordings:
            by_experiment[rec.experiment_index].append(rec)

        for exp_ix in sorted(by_experiment):
            print(f"    Experiment {exp_ix}")
            for rec in sorted(by_experiment[exp_ix], key=lambda r: r.recording_index):
                print(f"      Recording {rec.recording_index}  [recording_index={rec.recording_index}]")
                for stream_ix, stream_name in enumerate(rec.continuous._names):
                    cont = rec.continuous[stream_ix]
                    n_samples, n_channels = cont.samples.shape
                    fs = cont.metadata.sample_rate
                    duration = n_samples / fs if fs else float("nan")
                    print(
                        f"        Stream {stream_ix}: {stream_name}"
                        f"  — {n_channels} ch,"
                        f" {n_samples:,} samples"
                        f" @ {fs} Hz,"
                        f" {duration:.3f} s"
                    )


def load_tetrode_dataarray(
    acq_dir: str | pathlib.Path,
    record_node_index: int = 0,
    experiment_index: int = 0,
    recording_index: int = 0,
    stream_index: int = 0,
    *,
    selected_tetrodes: list[int] | None = None,
    layout: Layout = "2d_multi_index",
) -> xr.DataArray:
    """Load a tetrode-shaped DataArray from one continuous stream.

    Parameters
    ----------
    acq_dir
        Open Ephys acquisition directory.
    record_node_index, experiment_index, recording_index, stream_index
        Positional selectors matching :func:`print_session_tree` output.
    selected_tetrodes
        1-indexed tetrode numbers to include, in the desired output order.
        ``None`` loads all tetrodes in channel-map order.
    layout
        Output layout:

        - ``"2d_flat_index"``: dims ``(time, channel)`` with ``tetrode``
          and ``oe_index`` as 1-D non-index coords.
        - ``"2d_multi_index"`` (default): dims ``(time, electrode)`` with
          ``electrode`` as a pandas MultiIndex of ``(tetrode, channel)``.
        - ``"3d"``: dims ``(time, tetrode, channel)``; per-tetrode views
          are contiguous.

    All layouts carry ``attrs={"fs": ..., "units": "microvolts"}``.
    """
    acq_dir = pathlib.Path(acq_dir)
    session = oea.Session(str(acq_dir))
    node = session.recordnodes[record_node_index]

    channel_map = _load_channel_map(pathlib.Path(node.directory) / "settings.xml")
    tt2ch_ixs = channel_map["tt2ch_ixs"]
    tt2ch_names = channel_map["tt2ch_names"]

    rec = _get_recording(node, experiment_index, recording_index)

    if not 0 <= stream_index < len(rec.continuous._names):
        raise ValueError(
            f"stream_index={stream_index} out of range; recording has "
            f"{len(rec.continuous._names)} streams: {rec.continuous._names}"
        )
    cont = rec.continuous[stream_index]
    fs = cont.metadata.sample_rate

    if selected_tetrodes is None:
        selected_tetrodes = sorted(tt2ch_ixs)

    oe_ixs: list[int] = []
    ch_names: list[int] = []
    tt_labels: list[int] = []
    for tt in selected_tetrodes:
        if tt not in tt2ch_ixs:
            raise ValueError(
                f"Tetrode {tt} not in channel map (available: {sorted(tt2ch_ixs)})"
            )
        oe_ixs.extend(tt2ch_ixs[tt])
        ch_names.extend(tt2ch_names[tt])
        tt_labels.extend([tt] * 4)

    ns = cont.samples.shape[0]
    samples = cont.get_samples(
        start_sample_index=0,
        end_sample_index=ns,
        selected_channels=oe_ixs,
    )
    timestamps = cont.timestamps
    attrs = {"fs": fs, "units": "microvolts"}

    if layout == "2d_flat_index":
        return xr.DataArray(
            samples,
            dims=("time", "channel"),
            coords={
                "time": ("time", timestamps),
                "channel": ("channel", ch_names),
                "tetrode": ("channel", tt_labels),
                "oe_index": ("channel", oe_ixs),
            },
            attrs=attrs,
        )

    if layout == "2d_multi_index":
        midx = pd.MultiIndex.from_arrays(
            [tt_labels, ch_names], names=("tetrode", "channel")
        )
        return xr.DataArray(
            samples,
            dims=("time", "electrode"),
            coords=xr.Coordinates.from_pandas_multiindex(midx, "electrode"),
            attrs=attrs,
        ).assign_coords(time=("time", timestamps))

    if layout == "3d":
        n_tt = len(selected_tetrodes)
        samples_3d = samples.reshape(ns, n_tt, 4)  # view, no copy
        tt_names_2d = np.array(tt_labels, dtype=np.int64).reshape(n_tt, 4)
        ch_names_2d = np.array(ch_names, dtype=np.int64).reshape(n_tt, 4)
        oe_ixs_2d = np.array(oe_ixs, dtype=np.int64).reshape(n_tt, 4)
        return xr.DataArray(
            samples_3d,
            dims=("time", "tetrode", "channel"),
            coords={
                "time": ("time", timestamps),
                "tetrode": ("tetrode", list(range(n_tt))),
                "channel": ("channel", list(range(4))),
                "tetrode_name": (("tetrode", "channel"), tt_names_2d),
                "channel_name": (("tetrode", "channel"), ch_names_2d),
                "oe_index": (("tetrode", "channel"), oe_ixs_2d),
            },
            attrs=attrs,
        )

    raise ValueError(
        f"Unknown layout {layout!r}; expected one of "
        "'2d_flat_index', '2d_multi_index', '3d'."
    )

"""Loading helpers for Open Ephys tetrode acquisitions."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Literal

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from open_ephys import analysis as oea


Layout = Literal["2d_flat_index", "2d_multi_index", "3d"]


def _parse_channel_map_xml(settings_path: str | pathlib.Path) -> dict:
    """Parse a record node's ``settings.xml`` Channel Map processor.

    Returns a dict with the channel order found in the Channel Map, the
    per-channel enabled flags, and the tetrode → OE-channel-index groupings
    (``tt2ch_ixs``) derived from contiguous groups of four. Tetrode numbers are
    1-indexed so callers can refer to physical tetrodes by their natural labels.

    The ``index`` values are 0-based positions into the ``structure.oebin``
    channel order (i.e. into a recording's ``channel_names`` / a SpikeInterface
    extractor's ``channel_ids``).
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

    return {
        "oe_indices": oe_indices,
        "enabled": enabled,
        "tt2ch_ixs": tt2ch_ixs,
    }


def _settings_filename(experiment_name: str) -> str:
    """Map an Open Ephys ``experimentN`` folder name to its settings file.

    Open Ephys (and neo) name the first experiment's settings ``settings.xml``
    and every later experiment's ``settings_{N}.xml``.
    """
    n = int(experiment_name.removeprefix("experiment"))
    return "settings.xml" if n == 1 else f"settings_{n}.xml"


def _settings_filename_for_block(
    node_dir: pathlib.Path, block_index: int | None
) -> str:
    """Pick the settings filename for the ``block_index``-th experiment in a node."""
    exp_dirs = sorted(
        (
            d
            for d in node_dir.iterdir()
            if d.is_dir() and d.name.startswith("experiment")
        ),
        key=lambda d: int(d.name.removeprefix("experiment")),
    )
    bi = block_index or 0
    if not exp_dirs:
        return "settings.xml"
    if not 0 <= bi < len(exp_dirs):
        raise ValueError(
            f"block_index={bi} out of range; {node_dir} has experiments "
            f"{[d.name for d in exp_dirs]}"
        )
    return _settings_filename(exp_dirs[bi].name)


def _extractor_node_name(extractor, folder_structure: dict) -> str:
    """Resolve which record node in ``folder_structure`` the extractor belongs to."""
    if len(folder_structure) == 1:
        return next(iter(folder_structure))
    stream_name = getattr(extractor, "stream_name", "") or ""
    node_name = stream_name.split("#")[0] if "#" in stream_name else ""
    if node_name in folder_structure:
        return node_name
    raise ValueError(
        f"Could not match extractor stream {stream_name!r} to a record node; "
        f"available nodes: {list(folder_structure)}"
    )


def _settings_path_from_extractor(
    extractor, block_index: int | None
) -> pathlib.Path:
    """Read the per-experiment ``settings_file`` neo stored on the extractor.

    SpikeInterface's ``OpenEphysBinaryRecordingExtractor`` wraps neo's
    ``OpenEphysBinaryRawIO``, whose ``folder_structure`` carries the
    ``settings_file`` path for each experiment. We pick the experiment matching
    the extractor's block (``block_index``), so no path guessing is needed.
    """
    folder_structure = extractor.neo_reader.folder_structure
    node_name = _extractor_node_name(extractor, folder_structure)
    experiments = folder_structure[node_name]["experiments"]
    exp_ids = sorted(experiments)  # int keys → natural experiment order
    bi = block_index if block_index is not None else getattr(extractor, "block_index", 0)
    if not 0 <= bi < len(exp_ids):
        raise ValueError(
            f"block_index={bi} out of range; node {node_name!r} has "
            f"{len(exp_ids)} experiment(s): "
            f"{[experiments[e]['name'] for e in exp_ids]}"
        )
    return pathlib.Path(experiments[exp_ids[bi]]["settings_file"])


def _resolve_settings_path(source, block_index: int | None) -> pathlib.Path:
    """Resolve a channel-map ``source`` to a ``settings.xml`` path.

    ``source`` may be an OpenEphys extractor (uses neo's stored ``settings_file``),
    a path to a ``settings*.xml`` file, an acquisition directory, or a record-node
    directory.
    """
    if hasattr(source, "neo_reader"):
        return _settings_path_from_extractor(source, block_index)
    p = pathlib.Path(source)
    if p.is_file():
        return p
    if not p.is_dir():
        raise ValueError(f"{source!r} is not an extractor, file, or directory")
    node_dirs = sorted(p.glob("Record Node*"))
    if node_dirs:  # acquisition directory
        node_dir = node_dirs[0]
    elif list(p.glob("settings*.xml")):  # record-node directory
        node_dir = p
    else:
        raise ValueError(
            f"{p} is not an Open Ephys acquisition or record-node directory "
            "(no 'Record Node*' subdir and no settings*.xml)"
        )
    return node_dir / _settings_filename_for_block(node_dir, block_index)


def load_channel_map(source, *, block_index: int | None = None) -> dict:
    """Load the tetrode channel map for an Open Ephys acquisition.

    Parameters
    ----------
    source
        Any of: a SpikeInterface ``OpenEphysBinaryRecordingExtractor`` (the
        channel map is read from the ``settings_file`` neo stored on it), a path
        to a ``settings*.xml`` file, an acquisition directory, or a record-node
        directory.
    block_index
        For multi-experiment recordings, which experiment (0-based) to read the
        map for. Defaults to the extractor's own ``block_index`` (or the first
        experiment for directory inputs).

    Returns
    -------
    dict
        ``{"oe_indices", "enabled", "tt2ch_ixs"}`` — see
        :func:`_parse_channel_map_xml`.

    Notes
    -----
    The channel map (tetrode grouping) is an experiment-level property
    (``settings.xml``), whereas channel *names* are a per-recording property
    (each recording's ``structure.oebin``). neo/SpikeInterface only enforce
    consistency of stream names across a block's recordings and expose
    segment-0's channels; this loader follows that same convention.
    """
    return _parse_channel_map_xml(_resolve_settings_path(source, block_index))


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
    Stream indices match what :func:`open_tetrode_dataarray` expects.
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
            recs = sorted(by_experiment[exp_ix], key=lambda r: r.recording_index)
            exp_name = pathlib.Path(recs[0].directory).parent.name
            print(f"    {exp_name}  [experiment_index={exp_ix}]")
            for rec in recs:
                rec_name = pathlib.Path(rec.directory).name
                print(f"      {rec_name}  [recording_index={rec.recording_index}]")
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


def open_tetrode_dataarray(
    acq_dir: str | pathlib.Path,
    record_node_index: int = 0,
    experiment_index: int = 0,
    recording_index: int = 0,
    stream_index: int = 0,
    *,
    selected_tetrodes: list[int] | None = None,
    layout: Layout = "2d_multi_index",
    dtype: Literal["float32", "float64"] = "float32",
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

        - ``"2d_flat_index"``: dims ``(time, channel)`` with ``tetrode`` as a
          1-D non-index coord. ``channel`` carries the Open Ephys channel names
          (e.g. ``"CH40"``) and ``tetrode`` the labels ``"TT1"``, ``"TT2"``, …
        - ``"2d_multi_index"`` (default): dims ``(time, electrode)`` with
          ``electrode`` as a pandas MultiIndex of ``(tetrode, channel)``.
        - ``"3d"``: dims ``(time, tetrode, channel)``; per-tetrode views
          are contiguous.
    dtype
        Floating dtype of the (lazy) traces: ``"float32"`` (default) or
        ``"float64"``. The conversion stays in this dtype end to end — no
        ``float64`` intermediate when ``"float32"`` is requested.

    Channel names are taken from each recording's ``structure.oebin`` (a
    per-recording property), while the tetrode grouping comes from the selected
    experiment's ``settings.xml`` Channel Map (an experiment-level property).
    All layouts carry ``attrs={"fs": ..., "units": "microvolts"}``.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(
            f"dtype must be 'float32' or 'float64', got {dtype!r}"
        )
    acq_dir = pathlib.Path(acq_dir)
    session = oea.Session(str(acq_dir))
    node = session.recordnodes[record_node_index]

    rec = _get_recording(node, experiment_index, recording_index)

    # The channel map is an experiment-level property; use the settings file for
    # the selected experiment (settings.xml for experiment1, settings_N.xml
    # otherwise) rather than always reading experiment1's.
    exp_name = pathlib.Path(rec.directory).parent.name
    settings_path = pathlib.Path(node.directory) / _settings_filename(exp_name)
    channel_map = _parse_channel_map_xml(settings_path)
    tt2ch_ixs = channel_map["tt2ch_ixs"]

    if not 0 <= stream_index < len(rec.continuous._names):
        raise ValueError(
            f"stream_index={stream_index} out of range; recording has "
            f"{len(rec.continuous._names)} streams: {rec.continuous._names}"
        )
    cont = rec.continuous[stream_index]
    fs = cont.metadata.sample_rate

    if selected_tetrodes is None:
        selected_tetrodes = sorted(tt2ch_ixs)

    all_names = cont.metadata.channel_names  # from structure.oebin, OE order
    oe_ixs: list[int] = []
    ch_names: list[str] = []
    tt_labels: list[str] = []
    for tt in selected_tetrodes:
        if tt not in tt2ch_ixs:
            raise ValueError(
                f"Tetrode {tt} not in channel map (available: {sorted(tt2ch_ixs)})"
            )
        ch_ixs = tt2ch_ixs[tt]
        oe_ixs.extend(ch_ixs)
        ch_names.extend(all_names[ch] for ch in ch_ixs)  # e.g. "CH40"
        tt_labels.extend([f"TT{tt}"] * len(ch_ixs))  # e.g. "TT1"

    ns = cont.samples.shape[0]
    # Lazily wrap the int16 memmap: chunk along time, keep all channels in one
    # chunk so the 3d reshape below needs no rechunk.
    samples_int = da.from_array(cont.samples, chunks=("auto", -1))
    samples_int = samples_int[:, oe_ixs]  # select + reorder channels
    # bit_volts carries the target dtype so int16 → dtype scaling stays in
    # dtype (no float64 intermediate when dtype="float32").
    bit_volts = np.array(
        [cont.metadata.bit_volts[ch] for ch in oe_ixs], dtype=dtype
    )
    samples = samples_int.astype(dtype) * bit_volts  # → microvolts, lazy
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
        samples_3d = samples.reshape(ns, n_tt, 4)  # lazy reshape, no rechunk
        tt_names_2d = np.array(tt_labels).reshape(n_tt, 4)
        ch_names_2d = np.array(ch_names).reshape(n_tt, 4)
        return xr.DataArray(
            samples_3d,
            dims=("time", "tetrode", "channel"),
            coords={
                "time": ("time", timestamps),
                "tetrode": ("tetrode", list(range(n_tt))),
                "channel": ("channel", list(range(4))),
                "tetrode_name": (("tetrode", "channel"), tt_names_2d),
                "channel_name": (("tetrode", "channel"), ch_names_2d),
            },
            attrs=attrs,
        )

    raise ValueError(
        f"Unknown layout {layout!r}; expected one of "
        "'2d_flat_index', '2d_multi_index', '3d'."
    )


def build_tetrode_probegroup(
    n_tetrodes: int, *, inter_tetrode_um: float = 300.0
):
    """Build a :class:`probeinterface.ProbeGroup` of ``n_tetrodes`` tetrodes.

    Each tetrode is a :func:`probeinterface.generate_tetrode` probe, shifted
    ``inter_tetrode_um`` microns along x so the tetrodes are spatially separated
    (helps plotting; relative position does not affect per-group sorting). The
    contacts are wired identity (contact *i* → channel *i*), so the recording
    passed to :meth:`set_probegroup` must already present its channels in
    tetrode-contiguous, channel-map order — which :func:`attach_tetrode_probegroup`
    guarantees.
    """
    from probeinterface import ProbeGroup, generate_tetrode

    probegroup = ProbeGroup()
    for k in range(n_tetrodes):
        probe = generate_tetrode()
        probe.move([k * inter_tetrode_um, 0.0])
        probe.create_auto_shape()
        probegroup.add_probe(probe)
    probegroup.set_global_device_channel_indices(np.arange(4 * n_tetrodes))
    return probegroup


def attach_tetrode_probegroup(
    recording,
    channel_map: dict | None = None,
    *,
    selected_tetrodes: list[int] | None = None,
    geometry: bool = True,
    inter_tetrode_um: float = 300.0,
):
    """Return ``recording`` reordered and grouped to reflect the tetrode map.

    A raw OpenEphys SpikeInterface extractor exposes a flat list of channels with
    no probe and no grouping. This applies the same channel map / tetrode grouping
    that :func:`open_tetrode_dataarray` uses, so the recording can be split, sorted,
    and processed per tetrode (see SpikeInterface's "Work with tetrodes" and
    "Process a recording by channel group" how-tos).

    The returned recording has its channels reordered into channel-map order
    (tetrode-contiguous, e.g. ``CH40, CH38, CH36, CH34`` for ``TT1``) and carries:

    - a ``group`` property (0-based tetrode index) — enables
      ``recording.split_by("group")``, per-group preprocessing, and sort-by-group;
    - a ``tetrode`` property with the readable ``"TT1"``…``"TTn"`` labels matching
      the :func:`open_tetrode_dataarray` DataArray (``group`` ``g`` ↔ ``TT{g+1}``);
    - when ``geometry=True`` (default), a :class:`probeinterface.ProbeGroup` with
      one generic tetrode probe per group (synthetic geometry).

    Parameters
    ----------
    recording
        A SpikeInterface recording. If ``channel_map`` is ``None`` it must be a raw
        OpenEphys extractor (with a ``neo_reader``) so the map can be loaded.
    channel_map
        A map from :func:`load_channel_map`. If ``None``, loaded from ``recording``.
    selected_tetrodes
        1-indexed tetrode numbers to include, in the desired output order.
        ``None`` includes all tetrodes in channel-map order.
    geometry
        If ``True`` (default), attach a ProbeGroup (geometry + ``group`` property)
        via ``set_probegroup``. If ``False``, set only the ``group`` property
        (no probe), via ``set_property``.
    inter_tetrode_um
        Spatial offset between successive tetrode probes (only used when
        ``geometry=True``).
    """
    if channel_map is None:
        if not hasattr(recording, "neo_reader"):
            raise ValueError(
                "channel_map=None requires a raw OpenEphys extractor (with a "
                "neo_reader); pass an explicit channel_map from load_channel_map()."
            )
        channel_map = load_channel_map(recording)
    tt2ch_ixs = channel_map["tt2ch_ixs"]
    order = sorted(tt2ch_ixs) if selected_tetrodes is None else list(selected_tetrodes)

    n_channels = recording.get_num_channels()
    channel_ids = list(recording.get_channel_ids())
    flat_ixs: list[int] = []
    tt_labels: list[str] = []
    for tt in order:
        if tt not in tt2ch_ixs:
            raise ValueError(
                f"Tetrode {tt} not in channel map (available: {sorted(tt2ch_ixs)})"
            )
        ixs = tt2ch_ixs[tt]
        # Guard the channel-name consistency assumption: the map's OE indices must
        # index into the recording's (segment-0) channels.
        bad = [ix for ix in ixs if not 0 <= ix < n_channels]
        if bad:
            raise ValueError(
                f"Channel map references channel index/indices {bad} for tetrode "
                f"{tt}, but the recording has only {n_channels} channels — the "
                "channel map and recording are inconsistent."
            )
        flat_ixs.extend(ixs)
        tt_labels.extend([f"TT{tt}"] * len(ixs))

    # Reorder the recording into channel-map order; this *is* the channel map.
    ordered_ids = [channel_ids[ix] for ix in flat_ixs]
    ordered = recording.select_channels(ordered_ids)

    if geometry:
        probegroup = build_tetrode_probegroup(
            len(order), inter_tetrode_um=inter_tetrode_um
        )
        grouped = ordered.set_probegroup(probegroup, group_mode="by_probe")
    else:
        groups = np.repeat(np.arange(len(order)), 4)
        ordered.set_property("group", groups)
        grouped = ordered

    grouped.set_property("tetrode", np.asarray(tt_labels))
    return grouped

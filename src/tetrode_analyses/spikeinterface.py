"""SpikeInterface integration for Open Ephys tetrode sessions.

`get_recording` builds a SpikeInterface recording for all (default) or part of
an Open Ephys session, concatenating experiments (neo blocks) and recordings
(neo segments) into a single continuous recording whose time vector is the real
per-sample Open Ephys sync clock (preserving wall-clock gaps between experiments
and recordings). It attaches the tetrode ProbeGroup and `group`/`tetrode`
properties, and annotates provenance (a slice table + each experiment's settings
XML), analogous to `ecephys.wne.sglx.spikeinterface.get_recording`.

`convert_recording` compresses such a recording to a tetrode-aligned Zarr store
(see analyses/tetrode_preprocessing_and_sorting/compression for the benchmark
behind the scheme).
"""

from __future__ import annotations

import pathlib
import re
from typing import Literal

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from numcodecs import Blosc, Delta
from wavpack_numcodecs import WavPack

from tetrode_analyses.io import (
    _parse_channel_map_xml,
    _resolve_settings_path,
    attach_tetrode_probegroup,
)

Compressor = Literal["wavpack", "blosc-zstd"]

# Provenance columns annotated onto the recording (mirrors SGLX provenance).
# Times are SESSION-RELATIVE seconds: each (experiment, recording) segment is
# anchored by its sync_messages.txt "Software Time" (ms since Unix epoch), then
# the first segment's anchor is subtracted so the session starts at 0. Real
# wall-clock gaps between experiments and recordings are preserved. The raw
# Open Ephys board-clock endpoints are kept (oe_t_start/oe_t_end) for provenance.
_SLICE_TABLE_COLS = [
    "oe_experiment_index",
    "oe_recording_index",
    "oe_experiment_name",
    "oe_recording_name",
    "oe_settings_file",
    "fs",
    "n_samples",
    "start_sample",
    "end_sample",
    "oe_software_time_s",
    "oe_t_start",
    "oe_t_end",
    "t_start",
    "t_end",
    "gap_s",
]


def _read_software_time_s(recording_dir: pathlib.Path) -> float:
    """Parse the absolute start time (Unix-epoch seconds) from sync_messages.txt.

    The line is e.g. ``Software Time (milliseconds since midnight Jan 1st 1970
    UTC): 1779890872188``.
    """
    text = (recording_dir / "sync_messages.txt").read_text()
    m = re.search(r"Software Time[^:]*:\s*(\d+)", text)
    if not m:
        raise ValueError(f"No 'Software Time' line in {recording_dir}/sync_messages.txt")
    return int(m.group(1)) / 1000.0


def _num_blocks(oe_session_dir: pathlib.Path, stream_id: str) -> int:
    """Number of Open Ephys experiments (neo blocks) for this stream."""
    # A throwaway extractor on block 0 exposes the full block list via neo.
    ext = se.read_openephys(str(oe_session_dir), stream_id=stream_id, block_index=0)
    fs = ext.neo_reader.folder_structure
    node = next(iter(fs))
    return len(fs[node]["experiments"])


def _segment_records(
    oe_session_dir: pathlib.Path,
    stream_id: str,
    oe_experiment_index: int | None,
) -> tuple[list, list[np.ndarray], list[dict]]:
    """Enumerate (experiment, recording) segments in chronological order.

    Returns parallel lists of single-segment recordings, their per-sample time
    vectors (real Open Ephys sync clock), and per-segment metadata dicts. Time
    vectors are loaded eagerly via ``load_sync_timestamps=True``.
    """
    if oe_experiment_index is None:
        blocks = range(_num_blocks(oe_session_dir, stream_id))
    else:
        blocks = [oe_experiment_index]

    recs: list = []
    times: list[np.ndarray] = []
    meta: list[dict] = []
    for b in blocks:
        ext = se.read_openephys(
            str(oe_session_dir),
            stream_id=stream_id,
            block_index=b,
            load_sync_timestamps=True,
        )
        settings_path = _resolve_settings_path(ext, block_index=b)
        node_dir = settings_path.parent
        fs_struct = ext.neo_reader.folder_structure
        node = next(iter(fs_struct))
        exp_ids = sorted(fs_struct[node]["experiments"])
        exp_entry = fs_struct[node]["experiments"][exp_ids[b]]
        exp_name = exp_entry["name"]
        rec_entries = exp_entry["recordings"]
        rec_ids = sorted(rec_entries)
        for s in range(ext.get_num_segments()):
            seg = ext.select_segments([s])
            t = np.asarray(ext.get_times(segment_index=s))
            rec_name = (
                rec_entries[rec_ids[s]]["name"] if s < len(rec_ids) else f"recording{s + 1}"
            )
            recording_dir = node_dir / exp_name / rec_name
            recs.append(seg)
            times.append(t)
            meta.append(
                {
                    "oe_experiment_index": b,
                    "oe_recording_index": s,
                    "oe_experiment_name": exp_name,
                    "oe_recording_name": rec_name,
                    "oe_settings_file": str(settings_path),
                    "oe_software_time_s": _read_software_time_s(recording_dir),
                    "fs": float(ext.get_sampling_frequency()),
                    "n_samples": int(t.shape[0]),
                }
            )
    return recs, times, meta


def _session_relative_t0(meta: list[dict]) -> list[float]:
    """Session-relative start time (s) of each segment, gap-preserving.

    ``t0_seg = oe_software_time_seg - oe_software_time_first`` — the segment's
    start on a continuous wall clock zeroed at the first segment. Within-segment
    elapsed time (``oe_ts - oe_ts[0]``) is added per sample downstream.
    """
    ref = meta[0]["oe_software_time_s"]
    return [m["oe_software_time_s"] - ref for m in meta]


def _concat_session_times(times: list[np.ndarray], meta: list[dict]) -> np.ndarray:
    """Full session-relative time vector for the concatenated recording."""
    t0s = _session_relative_t0(meta)
    out = np.empty(sum(t.shape[0] for t in times), dtype="float64")
    cursor = 0
    for t, t0 in zip(times, t0s):
        n = t.shape[0]
        out[cursor : cursor + n] = t0 + (t - t[0])  # zero-based within segment
        cursor += n
    return out


def _build_slice_table(times: list[np.ndarray], meta: list[dict]) -> pd.DataFrame:
    """Assemble the slice table (session-relative times) from segments + metadata."""
    t0s = _session_relative_t0(meta)
    rows = []
    cursor = 0
    prev_t_end = None
    for t, m, t0 in zip(times, meta, t0s):
        n = m["n_samples"]
        oe_t_start = float(t[0])
        oe_t_end = float(t[-1])
        t_start = t0  # session-relative segment start (zero-based within segment)
        t_end = t0 + (oe_t_end - oe_t_start)
        rows.append(
            {
                **m,
                "start_sample": cursor,
                "end_sample": cursor + n,
                "oe_t_start": oe_t_start,
                "oe_t_end": oe_t_end,
                "t_start": t_start,
                "t_end": t_end,
                "gap_s": np.nan if prev_t_end is None else t_start - prev_t_end,
            }
        )
        cursor += n
        prev_t_end = t_end
    return pd.DataFrame(rows, columns=_SLICE_TABLE_COLS)


def make_oe_slice_table(
    oe_session_dir: str | pathlib.Path,
    oe_experiment_index: int | None = None,
    oe_stream_index: int = 0,
) -> pd.DataFrame:
    """Build a slice table for an Open Ephys session (one row per segment).

    Columns: experiment/recording indices and names, settings file, sampling
    rate, sample count, cumulative ``start_sample``/``end_sample`` in the
    concatenated recording, segment ``t_start``/``t_end`` (sync clock), and the
    ``gap_s`` to the previous segment (NaN for the first). Reflects real
    wall-clock gaps between experiments and between recordings.
    """
    oe_session_dir = pathlib.Path(oe_session_dir)
    _, times, meta = _segment_records(
        oe_session_dir, f"{oe_stream_index}", oe_experiment_index
    )
    return _build_slice_table(times, meta)


def _check_settings_match(slice_table: pd.DataFrame) -> None:
    """Assert the channel map is identical across all concatenated experiments."""
    settings_files = slice_table["oe_settings_file"].unique()
    maps = {f: _parse_channel_map_xml(f) for f in settings_files}
    ref_file, ref = next(iter(maps.items()))
    for f, m in maps.items():
        if m != ref:
            raise ValueError(
                "Channel maps differ across experiments being concatenated:\n"
                f"  {ref_file}\n  {f}\nRefusing to concatenate."
            )


def get_recording(
    oe_session_dir: str | pathlib.Path,
    oe_experiment_index: int | None = None,
    oe_stream_index: int = 0,
    inter_tetrode_um: float = 300.0,
    *,
    selected_tetrodes: list[int] | None = None,
) -> tuple[object, pd.DataFrame]:
    """Get a SpikeInterface recording for all (default) or part of an OE session.

    Parameters
    ----------
    oe_session_dir
        Open Ephys acquisition/session directory (``acq_dir``).
    oe_experiment_index
        Open Ephys experiment index = SI ``block_index``. ``None`` (default)
        concatenates every experiment (and every recording within them) into one
        continuous recording, like
        ``ecephys.wne.sglx.spikeinterface.get_recording``.
    oe_stream_index
        Open Ephys stream index; SI ``stream_id = f"{oe_stream_index}"``.
    inter_tetrode_um
        Spacing between successive tetrode probes in the attached ProbeGroup.
    selected_tetrodes
        1-indexed tetrodes to keep (in order). ``None`` keeps all. Use to drop
        bad tetrodes.

    Returns
    -------
    (recording, slice_table)
        ``recording`` has the real OE sync-clock time vector
        (``has_time_vector() is True``), tetrode-contiguous channel order,
        ``group``/``tetrode`` properties, and a tetrode ProbeGroup. It is
        annotated with ``openephys_provenance`` (the slice table) and
        ``openephys_settings`` (each experiment's settings XML).
    """
    oe_session_dir = pathlib.Path(oe_session_dir)
    stream_id = f"{oe_stream_index}"

    recs, times, meta = _segment_records(
        oe_session_dir, stream_id, oe_experiment_index
    )
    slice_table = _build_slice_table(times, meta)
    _check_settings_match(slice_table)

    # Concatenate segments into one continuous recording (ConcatenateSegmentRecording
    # ignores per-segment times), then set the real concatenated sync clock.
    base = (
        recs[0]
        if len(recs) == 1
        else si.ConcatenateSegmentRecording(recs, sampling_frequency_max_diff=1e-6)
    )

    # Channel map is identical across blocks (checked above); pass it explicitly
    # since a concatenated recording has no neo_reader for load_channel_map.
    channel_map = _parse_channel_map_xml(slice_table["oe_settings_file"].iloc[0])
    recording = attach_tetrode_probegroup(
        base,
        channel_map=channel_map,
        selected_tetrodes=selected_tetrodes,
        geometry=True,
        inter_tetrode_um=inter_tetrode_um,
    )

    # Set the session-relative, gap-preserving master time vector (sync_messages
    # anchored) on the FINAL recording: set_times does not propagate across the
    # channel reorder / set_probegroup that attach_tetrode_probegroup applies, so
    # this must come last.
    recording.set_times(_concat_session_times(times, meta), with_warning=False)
    if not recording.has_time_vector(segment_index=0):  # guard the ordering above
        raise RuntimeError("Time vector was not set on the recording.")

    settings_xml = {
        row.oe_experiment_name: pathlib.Path(row.oe_settings_file).read_text()
        for row in slice_table.drop_duplicates("oe_settings_file").itertuples()
    }
    recording.annotate(
        openephys_provenance=slice_table[_SLICE_TABLE_COLS].to_json(),
        openephys_settings=settings_xml,
    )
    return recording, slice_table


def _traces_compressor(compressor: Compressor, bps: float):
    if compressor == "wavpack":
        return WavPack(bps=bps) if bps > 0 else WavPack()
    if compressor == "blosc-zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    raise ValueError(f"Unknown compressor {compressor!r}; use 'wavpack' or 'blosc-zstd'.")


def convert_recording(
    recording,
    out_zarr: str | pathlib.Path,
    *,
    compressor: Compressor = "wavpack",
    bps: float = 2.25,
    time_chunk_s: float = 30.0,
    n_jobs: int = 16,
) -> object:
    """Compress a recording (from `get_recording`) to a tetrode-aligned Zarr store.

    ``compressor`` selects the *traces* codec (``wavpack`` => ``WavPack(bps)``,
    lossy unless ``bps=0``; ``blosc-zstd`` => lossless ``Blosc(zstd,5,BITSHUFFLE)``).
    Times and properties keep SI's lossless default, so the time vector is
    lossless in either store. Channels are chunked one tetrode (4 ch) per chunk.
    """
    out_zarr = pathlib.Path(out_zarr)
    traces_comp = _traces_compressor(compressor, bps)
    n_tt = recording.get_num_channels() // 4
    dur_h = recording.get_num_frames() / recording.sampling_frequency / 3600
    print(
        f"Converting {recording.get_num_channels()} ch / {n_tt} tetrodes, "
        f"{dur_h:.2f} h, traces={traces_comp}, channel_chunk=4, "
        f"chunk={time_chunk_s}s -> {out_zarr}"
    )
    return recording.save(
        format="zarr",
        folder=str(out_zarr),
        compressor_by_dataset={"traces": traces_comp},
        filters_by_dataset={"times": [Delta(dtype="float64")]},
        channel_chunk_size=4,
        chunk_duration=f"{time_chunk_s}s",
        n_jobs=n_jobs,
        progress_bar=True,
    )

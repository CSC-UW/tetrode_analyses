"""Convert an Open Ephys flat-binary tetrode recording to a tetrode-aligned,
compressed Zarr store optimized for spike sorting by tetrode group.

Scheme (see README.md for the benchmark evidence behind every choice):

  * Format        : Zarr (SpikeInterface ZarrRecordingExtractor, read-ready)
  * Channel chunk : 4  -> one tetrode per chunk. THE decisive parameter for
                    per-group sorting: a 64-ch chunk forces decompressing all
                    64 channels to read one tetrode (16x read amplification).
  * Probe / order : tetrode_analyses.io.attach_tetrode_probegroup() reorders
                    channels into tetrode-contiguous channel-map order (so
                    channel_chunk_size=4 aligns to one tetrode -- this rig's map
                    is NOT identity, e.g. TT1 = .dat cols 39,37,35,33), sets the
                    `group` (0-based) and `tetrode` ("TT1"...) properties, and
                    attaches a generic-tetrode ProbeGroup so the store is
                    sort-ready (mountainsort5 whitening needs channel locations).
  * Timestamps    : load_sync_timestamps=True -> the real per-sample Open Ephys
                    sync clock is carried as the time vector and written as a
                    Delta-compressed (lossless) `times_seg0` dataset. SI's
                    default (t_start + constant rate) is wrong for this rig.
  * Time chunk    : 30 s (900,000 samples). Read each tetrode in blocks >= the
                    time chunk (set the sorter's `chunk_duration`) so every
                    chunk is decoded once.
  * Compressor    : `wavpack` -> WavPack(bps=2.25) hybrid-lossy (7.1x here;
                    error ~5x below the 300-6000 Hz noise floor) -- matches the
                    lab's Neuropixel practice. `blosc-zstd` -> Blosc(zstd,5,
                    BITSHUFFLE), lossless (1.73x), SI's default. Only the traces
                    are (optionally) lossy; times/properties stay lossless.

Emit BOTH stores to compare lossless vs lossy sorting:

    cd gfys_workspace
    BASE=../tetrode_analyses/analyses/chunking_and_compression/06_convert.py
    ACQ=/Volumes/neuropixel_archive/tetrode_data/2026-05-27_09-07-52
    uv run python $BASE --acq $ACQ --out OUT/2026-05-27.exp1.wavpack-bps2.25.zarr \
        --block 0 --compressor wavpack --bps 2.25 --n-jobs 16
    uv run python $BASE --acq $ACQ --out OUT/2026-05-27.exp1.blosc-zstd.zarr \
        --block 0 --compressor blosc-zstd --n-jobs 16
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Literal

import spikeinterface.extractors as se
from numcodecs import Blosc, Delta
from wavpack_numcodecs import WavPack

from tetrode_analyses.io import attach_tetrode_probegroup

Compressor = Literal["wavpack", "blosc-zstd"]


def _traces_compressor(compressor: Compressor, bps: float):
    if compressor == "wavpack":
        return WavPack(bps=bps) if bps > 0 else WavPack()
    if compressor == "blosc-zstd":
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    raise ValueError(f"Unknown compressor {compressor!r}; use 'wavpack' or 'blosc-zstd'.")


def convert(
    acq_dir: str | pathlib.Path,
    out_zarr: str | pathlib.Path,
    *,
    stream_id: str = "0",
    stream_name: str | None = None,
    block_index: int = 0,
    compressor: Compressor = "wavpack",
    bps: float = 2.25,
    time_chunk_s: float = 30.0,
    inter_tetrode_um: float = 300.0,
    n_jobs: int = 16,
) -> object:
    """Convert one Open Ephys experiment (block) to a tetrode-aligned Zarr store.

    Parameters mirror the scheme in the module docstring. ``compressor`` selects
    the *traces* codec; times and properties always use SI's lossless default,
    so the time vector is lossless even when traces are lossy. ``bps`` applies
    only to ``compressor="wavpack"`` (0.0 => lossless WavPack).
    """
    acq_dir = pathlib.Path(acq_dir)
    out_zarr = pathlib.Path(out_zarr)

    # Real Open Ephys sync timestamps as the time vector (not t_start + fs).
    read_kwargs = dict(load_sync_timestamps=True, block_index=block_index)
    if stream_name is not None:
        read_kwargs["stream_name"] = stream_name
    else:
        read_kwargs["stream_id"] = stream_id
    rec = se.read_openephys(str(acq_dir), **read_kwargs)
    if not rec.has_time_vector(segment_index=0):
        raise RuntimeError(
            "Sync timestamps were not loaded; expected a per-sample time vector. "
            "Check that timestamps.npy exists for this stream."
        )

    # Reorder to tetrode-contiguous order + group/tetrode properties + ProbeGroup.
    rec = attach_tetrode_probegroup(
        rec, geometry=True, inter_tetrode_um=inter_tetrode_um
    )

    traces_comp = _traces_compressor(compressor, bps)
    n_tt = rec.get_num_channels() // 4
    print(
        f"Converting block {block_index}: {rec.get_num_channels()} ch / {n_tt} "
        f"tetrodes, {rec.get_num_frames() / rec.sampling_frequency / 3600:.2f} h, "
        f"traces={traces_comp}, channel_chunk=4, chunk={time_chunk_s}s -> {out_zarr}"
    )
    return rec.save(
        format="zarr",
        folder=str(out_zarr),
        compressor_by_dataset={"traces": traces_comp},
        filters_by_dataset={"times": [Delta(dtype="float64")]},
        channel_chunk_size=4,
        chunk_duration=f"{time_chunk_s}s",
        n_jobs=n_jobs,
        progress_bar=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acq", required=True, help="Open Ephys acquisition directory")
    p.add_argument("--out", required=True, help="Output .zarr path")
    p.add_argument("--stream-id", default="0", help="Neo stream id (ephys = '0')")
    p.add_argument("--stream-name", default=None, help="Overrides --stream-id if given")
    p.add_argument("--block", type=int, default=0, help="Experiment index (neo block)")
    p.add_argument("--compressor", choices=["wavpack", "blosc-zstd"], default="wavpack")
    p.add_argument("--bps", type=float, default=2.25, help="WavPack bits/sample; 0=lossless")
    p.add_argument("--time-chunk-s", type=float, default=30.0)
    p.add_argument("--inter-tetrode-um", type=float, default=300.0)
    p.add_argument("--n-jobs", type=int, default=16)
    a = p.parse_args()
    convert(
        a.acq, a.out, stream_id=a.stream_id, stream_name=a.stream_name,
        block_index=a.block, compressor=a.compressor, bps=a.bps,
        time_chunk_s=a.time_chunk_s, inter_tetrode_um=a.inter_tetrode_um,
        n_jobs=a.n_jobs,
    )


if __name__ == "__main__":
    main()

"""CLI: convert an Open Ephys tetrode session to a tetrode-aligned Zarr store.

Thin wrapper over `tetrode_analyses.spikeinterface.get_recording` +
`convert_recording` (see that module and README.md for the full scheme and the
benchmark evidence). Channels are reordered to tetrode-contiguous order with a
ProbeGroup attached, the real Open Ephys sync-clock time vector is preserved
(Delta-compressed, lossless), and traces are chunked one tetrode per chunk.

Emit BOTH stores (lossless + lossy) to compare sorting outcomes:

    cd gfys_workspace
    BASE=../tetrode_analyses/analyses/chunking_and_compression/06_convert.py
    ACQ=/Volumes/neuropixel_archive/tetrode_data/2026-05-27_09-07-52
    OUT=/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52
    uv run python $BASE --acq $ACQ --out $OUT/2026-05-27.wavpack-bps2.25.zarr \
        --compressor wavpack --bps 2.25 --n-jobs 32
    uv run python $BASE --acq $ACQ --out $OUT/2026-05-27.blosc-zstd.zarr \
        --compressor blosc-zstd --n-jobs 32

Omit --experiment to concatenate every experiment (the full session).
"""

from __future__ import annotations

import argparse

from tetrode_analyses.spikeinterface import convert_recording, get_recording


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acq", required=True, help="Open Ephys session directory")
    p.add_argument("--out", required=True, help="Output .zarr path")
    p.add_argument("--stream-index", type=int, default=0, help="OE stream index (ephys = 0)")
    p.add_argument(
        "--experiment", type=int, default=None,
        help="OE experiment index (SI block). Omit to concatenate all experiments.",
    )
    p.add_argument("--compressor", choices=["wavpack", "blosc-zstd"], default="wavpack")
    p.add_argument("--bps", type=float, default=2.25, help="WavPack bits/sample; 0=lossless")
    p.add_argument("--time-chunk-s", type=float, default=30.0)
    p.add_argument("--inter-tetrode-um", type=float, default=300.0)
    p.add_argument("--n-jobs", type=int, default=16)
    a = p.parse_args()

    recording, _slice_table = get_recording(
        a.acq,
        oe_experiment_index=a.experiment,
        oe_stream_index=a.stream_index,
        inter_tetrode_um=a.inter_tetrode_um,
    )
    convert_recording(
        recording, a.out, compressor=a.compressor, bps=a.bps,
        time_chunk_s=a.time_chunk_s, n_jobs=a.n_jobs,
    )


if __name__ == "__main__":
    main()

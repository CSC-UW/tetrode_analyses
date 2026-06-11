"""Download the viz bundle from tononi-2 to a local external drive via rclone.

RUN THIS ON YOUR LOCAL MACHINE (Mac/Linux), which has ``rclone`` configured with
a ``tononi-2`` remote rooted at the server filesystem (so server absolute paths
address as ``tononi-2:/nvme/...``)::

    python download_bundle.py --dest /Volumes/MyDrive/ttm_nod
    python download_bundle.py --dest /Volumes/MyDrive/ttm_nod --dry-run
    python download_bundle.py --dest /mnt/drive/ttm_nod --transfers 24

Stdlib only — no Python dependencies, just ``rclone`` on PATH. It copies the
*contents* of the server bundle into ``--dest`` (one folder), then you point
``launch_loupe.py --data-dir`` at that folder.

The bundle is a mix of many small zarr chunk files plus a single large
``spikes.parquet``. ``--transfers``/``--checkers`` parallelize the many-files
case; ``rclone`` auto-multithreads the large parquet. ``--no-check-dest`` (the
default here) skips per-file checksumming, which is ~15x faster than tar-pipe for
a first one-way copy (benchmarked in offproj's zarr staging). Pass ``--check``
for a resumable, checksum-verified re-run instead.

First build the bundle on tononi-2 with ``build_bundle.py``.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

DEFAULT_REMOTE = "tononi-2"
DEFAULT_SERVER_BUNDLE = (
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/viz_bundle"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        required=True,
        help="Local destination folder on the external drive (created if absent).",
    )
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="rclone remote name (default: tononi-2)."
    )
    parser.add_argument(
        "--server-bundle",
        default=DEFAULT_SERVER_BUNDLE,
        help="Absolute bundle path on the server (default: the TTM-NOD viz_bundle).",
    )
    parser.add_argument("--transfers", type=int, default=16)
    parser.add_argument("--checkers", type=int, default=16)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Checksum-verify (drop --no-check-dest) for resumable re-runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not shutil.which("rclone"):
        sys.exit(
            "rclone not found on PATH. Install it: https://rclone.org/install/"
        )

    source = f"{args.remote}:{args.server_bundle}"
    args.dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone",
        "copy",
        source,
        str(args.dest),
        f"--transfers={args.transfers}",
        f"--checkers={args.checkers}",
        "--progress",
    ]
    if not args.check:
        cmd.append("--no-check-dest")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Downloading {source} -> {args.dest}")
    print("  " + " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"rclone copy failed (exit {result.returncode}).")
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    print(f"\nLaunch the viewer:\n  uv run python launch_loupe.py --data-dir {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

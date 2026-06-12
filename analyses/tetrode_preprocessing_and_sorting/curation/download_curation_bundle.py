"""Download a curation bundle from tononi-2 to a local external drive via rclone.

RUN THIS ON YOUR LOCAL MACHINE (rclone configured with a ``tononi-2`` remote
rooted at the server filesystem). It first pulls the small manifest written by
``build_curation_manifest.py``, then rclones each item it lists to ``--dest``,
preserving the analyzer→recording relative layout so ``launch_curation_local.py``
can auto-resolve the recording::

    python download_curation_bundle.py --dest /Volumes/MyDrive/ttm_nod_curation
    python download_curation_bundle.py --dest /Volumes/MyDrive/ttm_nod_curation --no-traces
    python download_curation_bundle.py --dest /Volumes/MyDrive/ttm_nod_curation --dry-run

Stdlib only — just ``rclone`` on PATH. The recording is hundreds of GB; pass
``--no-traces`` (or use a ``--no-traces`` manifest) to fetch only the ~1.7 GB
analyzer and curate without the raw-traces view. ``--no-check-dest`` (default) is
fastest for a first one-way copy; ``--check`` re-enables checksumming for
resumable re-runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

DEFAULT_REMOTE = "tononi-2"
DEFAULT_MANIFEST = (
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/"
    "curation_bundles/tracked_48h.manifest.json"
)


def rclone(args_list: list[str]) -> None:
    result = subprocess.run(["rclone", *args_list])
    if result.returncode != 0:
        sys.exit(f"rclone failed (exit {result.returncode}): rclone {' '.join(args_list)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=pathlib.Path, required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help="Absolute manifest path on the server (from build_curation_manifest.py).",
    )
    parser.add_argument(
        "--no-traces", action="store_true",
        help="Skip the recording item even if the manifest includes it.",
    )
    parser.add_argument("--transfers", type=int, default=16)
    parser.add_argument("--checkers", type=int, default=16)
    parser.add_argument(
        "--check", action="store_true",
        help="Checksum-verify (drop --no-check-dest) for resumable re-runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not shutil.which("rclone"):
        sys.exit("rclone not found on PATH. Install it: https://rclone.org/install/")

    args.dest.mkdir(parents=True, exist_ok=True)

    # 1. Fetch the manifest (tiny — always real, even on --dry-run, so we know items).
    local_manifest = args.dest / "curation_manifest.json"
    print(f"Fetching manifest {args.remote}:{args.manifest}")
    rclone(["copyto", f"{args.remote}:{args.manifest}", str(local_manifest)])
    manifest = json.loads(local_manifest.read_text())
    skip_recording = args.no_traces or not manifest.get("with_traces", True)

    # 2. Plan + copy each item.
    to_copy = [
        it for it in manifest["items"]
        if not (it["role"] == "recording" and skip_recording)
    ]
    print(f"\nBundle: {manifest['subject']}/{manifest['experiment']} "
          f"sort={manifest['sorting']} (with_traces={not skip_recording})")
    for it in to_copy:
        print(f"  - {it['role']:9s} {it.get('size') or '?':>6}  -> {it['dest_relpath']}")
    if skip_recording and any(it["role"] == "recording" for it in manifest["items"]):
        print("  - recording  (skipped: --no-traces)")

    base_flags = [
        f"--transfers={args.transfers}", f"--checkers={args.checkers}", "--progress",
    ]
    if not args.check:
        base_flags.append("--no-check-dest")
    if args.dry_run:
        base_flags.append("--dry-run")

    t0 = time.perf_counter()
    for it in to_copy:
        src = f"{args.remote}:{it['server_path']}"
        dst = args.dest / it["dest_relpath"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {it['role']} -> {dst} ===")
        rclone(["copy", src, str(dst), *base_flags])

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
    print(f"\nLaunch curation:\n  uv run python launch_curation_local.py --data-dir {args.dest}"
          + (" --no-traces" if skip_recording else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

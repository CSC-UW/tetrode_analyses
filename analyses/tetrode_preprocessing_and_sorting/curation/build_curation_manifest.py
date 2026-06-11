"""Write a lightweight manifest for downloading + curating a sort locally.

RUN THIS ON tononi-2. Unlike the loupe viz bundle, curation data is far too large
to stage into one folder: the recording the sort was produced from is hundreds of
GB. So the "bundle" is just a JSON manifest listing the items curation needs and
their current locations on tononi-2; ``download_curation_bundle.py`` then rclones
each item straight from its real location to an external drive.

Curation needs only two things:

  * ``analyzer.zarr`` (~1.7 GB) — the SortingAnalyzer. It embeds its own sorting
    (so ``aggregated/`` and ``by_group/`` are not needed) and references its
    recording by a relative path.
  * the recording the sort was produced from — the ``blosc-zstd`` store
    (~352 GB), needed only for the **traces** view. Pass ``--no-traces`` to omit
    it (then curate metrics/waveforms/correlograms without raw traces).

The manifest records each item's ``dest_relpath`` so the download preserves the
analyzer→recording relative layout (analyzer at
``sortings_seed42_pcafix/<sorting>/analyzer.zarr``, recording at the drive root);
then ``load_sorting_analyzer`` auto-resolves the recording locally with no
reconstruction — bit-exact, the same data that was sorted.

    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/curation/build_curation_manifest.py
    uv run python ../tetrode_analyses/.../curation/build_curation_manifest.py --no-traces
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess

from tetrode_analyses import experiment as exp

SUBJECT, EXPERIMENT = "TTM-001", "TTM-NOD"
DEFAULT_SORTINGS_SUBDIR = "sortings_seed42_pcafix"
DEFAULT_SORTING = "blosc-43200s-train3600s"  # 48 h, 12 h blocks, 1 h training window
# The recording the sort was produced from (the analyzer references it by relative
# path). We deliberately support ONLY this store, never a smaller/lossy variant.
RECORDING_COMPRESSOR = "blosc-zstd"


def du_size(path: pathlib.Path, timeout: float = 120.0) -> str | None:
    """Best-effort human-readable size via ``du -sh`` (None on failure/timeout)."""
    try:
        out = subprocess.run(
            ["du", "-sh", str(path)],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        return out.stdout.split("\t", 1)[0].strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorting", default=DEFAULT_SORTING)
    parser.add_argument("--sortings-subdir", default=DEFAULT_SORTINGS_SUBDIR)
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Omit the recording; curate without the raw-traces view.",
    )
    parser.add_argument("--no-size", action="store_true", help="Skip du size lookup.")
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        default=None,
        help="Manifest path (default: <root>/curation_bundles/<sorting>.manifest.json).",
    )
    args = parser.parse_args()

    params = exp.load_experiment_params(
        exp.experiment_params_path(SUBJECT, EXPERIMENT)
    )
    root = pathlib.Path(params.openephys_session)
    session = root.name
    with_traces = not args.no_traces

    analyzer_path = root / args.sortings_subdir / args.sorting / "analyzer.zarr"
    if not analyzer_path.exists():
        raise FileNotFoundError(f"analyzer.zarr not found: {analyzer_path}")
    recording_path = root / f"{session}.{RECORDING_COMPRESSOR}.zarr"
    if with_traces and not recording_path.exists():
        raise FileNotFoundError(
            f"recording not found: {recording_path} (use --no-traces to skip traces)"
        )

    def item(role, path, dest_relpath, required):
        return {
            "role": role,
            "server_path": str(path),
            "dest_relpath": dest_relpath,
            "size": None if args.no_size else du_size(path),
            "required": required,
        }

    # dest_relpaths preserve the analyzer -> recording relative link: the analyzer
    # at sortings_subdir/<sorting>/analyzer.zarr references ../../../<rec>.zarr, so
    # the recording must land at the drive root for auto-resolution.
    items = [
        item(
            "analyzer", analyzer_path,
            f"{args.sortings_subdir}/{args.sorting}/analyzer.zarr", True,
        )
    ]
    if with_traces:
        items.append(
            item("recording", recording_path, recording_path.name, False)
        )

    manifest = {
        "kind": "curation_manifest",
        "subject": SUBJECT,
        "experiment": EXPERIMENT,
        "session": session,
        "sorting": args.sorting,
        "sortings_subdir": args.sortings_subdir,
        "recording_compressor": RECORDING_COMPRESSOR,
        "with_traces": with_traces,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "items": items,
    }

    out = args.manifest_out or (
        root / "curation_bundles" / f"{args.sorting}.manifest.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest -> {out}")
    print(
        "Download it with:\n"
        f"  python download_curation_bundle.py --manifest {out} "
        "--dest <external-drive>/ttm_nod_curation"
        + ("" if with_traces else "  (--no-traces manifest)")
    )


if __name__ == "__main__":
    main()

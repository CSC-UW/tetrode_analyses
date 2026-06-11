"""Launch spikeinterface-gui curation over a locally-downloaded curation bundle.

RUN THIS ON YOUR LOCAL MACHINE, pointing ``--data-dir`` at the folder
``download_curation_bundle.py`` populated::

    uv run python launch_curation_local.py --data-dir /Volumes/MyDrive/ttm_nod_curation
    uv run python launch_curation_local.py --data-dir ... --style grahams_curation
    uv run python launch_curation_local.py --data-dir ... --no-traces

Reads ``curation_manifest.json`` in ``--data-dir``, loads the analyzer, and runs
the GUI with curation enabled. For the **traces** view the recording is
auto-resolved from the preserved on-disk layout (the analyzer references it by a
relative path; the downloader placed the ``blosc-zstd`` store — the exact data the
sort was produced from — at the drive root). ``--no-traces`` curates without raw
traces (no recording needed).

Install: ``tetrode_analyses[curation]`` (spikeinterface-gui → spikeinterface).
``--mode desktop`` (default) is the Qt backend (a Mac has a display); ``--mode
web`` serves the Panel backend on ``localhost:<port>``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # import the sibling server launcher for styles
from launch_curation import STYLES, style_kwargs  # noqa: E402

MANIFEST_FILENAME = "curation_manifest.json"
DEFAULT_PORT = 8000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=pathlib.Path, required=True,
        help="Folder populated by download_curation_bundle.py.",
    )
    parser.add_argument("--mode", choices=["desktop", "web"], default="desktop")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SIGUI_PORT", DEFAULT_PORT)),
        help=f"Port for web mode (default {DEFAULT_PORT} or $SIGUI_PORT).",
    )
    parser.add_argument(
        "--style", choices=sorted(STYLES), default=None,
        help="Named layout+settings preset (e.g. 'grahams_curation').",
    )
    parser.add_argument(
        "--no-traces", action="store_true",
        help="Curate without the raw-traces view (no recording needed).",
    )
    args = parser.parse_args()

    manifest_path = args.data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILENAME} not found in {args.data_dir}. Download the bundle "
            "first with download_curation_bundle.py."
        )
    manifest = json.loads(manifest_path.read_text())
    analyzer_rel = next(
        it["dest_relpath"] for it in manifest["items"] if it["role"] == "analyzer"
    )
    analyzer_path = args.data_dir / analyzer_rel
    if not analyzer_path.exists():
        raise FileNotFoundError(
            f"analyzer.zarr not found at {analyzer_path}. Re-run "
            "download_curation_bundle.py."
        )

    want_traces = not args.no_traces and manifest.get("with_traces", True)

    # argparse has consumed our flags; clear sys.argv so Qt's QApplication (desktop
    # mode) doesn't reinterpret --style as a built-in widget-style option.
    sys.argv = sys.argv[:1]
    if args.mode == "web":
        os.environ.setdefault("BROWSER", "echo")  # never hand the URL to a TTY browser

    import spikeinterface as si
    import spikeinterface_gui as sg

    analyzer = si.load_sorting_analyzer(analyzer_path)

    if want_traces and not analyzer.has_recording():
        raise FileNotFoundError(
            "Traces requested but the recording did not auto-resolve next to the "
            f"analyzer ({manifest.get('recording_compressor', 'blosc-zstd')} store "
            "expected at the bundle root). Re-download with traces, or pass "
            "--no-traces to curate without the raw-traces view."
        )
    with_traces = want_traces
    print(
        f"Curation: {manifest['sorting']} | {analyzer.sorting.get_num_units()} units "
        f"| traces={'on' if with_traces else 'off'} | mode={args.mode}",
        flush=True,
    )

    skw = style_kwargs(args.style)
    if args.mode == "web":
        print(f"\nServing curation GUI at  http://localhost:{args.port}/   (Ctrl-C to stop)\n",
              flush=True)
        sg.run_mainwindow(
            analyzer, mode="web", curation=True, with_traces=with_traces,
            address="localhost", port=args.port,
            panel_start_server_kwargs={"show": False}, **skw,
        )
    else:
        print("Launching desktop (Qt) curation GUI.", flush=True)
        sg.run_mainwindow(
            analyzer, mode="desktop", curation=True, with_traces=with_traces, **skw
        )


if __name__ == "__main__":
    main()

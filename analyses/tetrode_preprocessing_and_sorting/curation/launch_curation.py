"""Launch the spikeinterface-gui curation GUI for the chunk-tracked 48 h sort.

`--mode web` (default) runs the Panel web backend, headless-safe: it does NOT
auto-open a browser (tononi-2 has no display, and the default `www-browser` is
w3m, which would hijack the terminal). Reach it from your laptop over an SSH
tunnel (printed at startup). `--mode desktop` runs the Qt backend, which needs a
display (run over `ssh -X` or VNC).

`--style grahams_curation` applies a named preset: a layout
(grahams_curation_layout.json) + per-view settings (grahams_curation_settings.json),
both next to this script. The unit-list columns are delivered via the
`displayed_unit_properties` param rather than the settings file, because
spikeinterface-gui's settings file cannot drive the unit-list columns -- see
gfys_workspace/docs/developer_notes/spikeinterface_gui_style_gaps.md for that and
the other capability gaps (trace default window, metrics selection, column order).

Web port is pinned (--port / $SIGUI_PORT, default 8000) so the tunnel is
predictable; address stays "localhost" so a same-port tunnel is websocket-origin
valid. Web-only options are not passed in desktop mode. The GUI blocks until you
quit it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Chunk-tracked 48 h sort, geometry-free QC analyzer (scripts 36/37; the recommended
# sorting). Earlier sorts kept here for reference:
#   .../sortings_seed42_pcafix/blosc-43200s-train3600s/analyzer.zarr   (12 h-block scheme3)
#   .../sortings_seed42_pcafix/blosc-scheme2-train3600s/analyzer.zarr  (48 h single-block)
ANALYZER_PATH = Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/"
    "sortings_seed42_pcafix/tracked_48h/analyzer_clustered.zarr"
)
DEFAULT_PORT = 8000

# Named curation presets: layout + per-view settings + unit-table columns.
# `displayed_unit_properties` carries the unit-list columns because the settings
# file cannot (the per-column setting tree does not drive the display).
STYLES = {
    "grahams_curation": {
        "layout_file": SCRIPT_DIR / "grahams_curation_layout.json",
        "settings_file": SCRIPT_DIR / "grahams_curation_settings.json",
        # `tier`/`n_chunks`/`track_hours` are persisted unit properties (script 45):
        # sort the unit list by `tier` (or `tier_level`) to group conservative/moderate/
        # permissive, or by `n_chunks`/`track_hours` for the longest-tracked units.
        # `unitrefine_neural_prob`/`unitrefine_label` are the ADVISORY noise/neural ranking
        # (script 50, tetrode_analyses.unitrefine_advisory): sort by `unitrefine_neural_prob`
        # to rank most-neural-first. ADVISORY ONLY -- the probability is uncalibrated on
        # tetrodes (~0.5), so rank by it; do not treat the 0.5 label as a hard cut. The rest
        # are the metrics driving the isolation gate (rp_contamination OR sliding_rp_violation
        # + firing-rate floor); see _track_eval.isolation_tier_mask / TRACKING_FINDINGS.md.
        "displayed_unit_properties": [
            "group",
            "tier",
            "n_chunks",
            "track_hours",
            "n_windows",          # matching-pursuit tracks (analyzer_tracks.zarr): windows present
            "identity_min_cos",   # matching-pursuit tracks: low (<0.7) = suspect drift/swap, inspect
            "unitrefine_label",
            "unitrefine_neural_prob",
            "firing_rate",
            "rp_contamination",
            "sliding_rp_violation",
        ],
    },
}


def style_kwargs(style_name):
    """run_mainwindow kwargs for a named style (empty for None)."""
    if style_name is None:
        return {}
    style = STYLES[style_name]
    kw = {
        "layout": json.loads(Path(style["layout_file"]).read_text()),
        "user_settings": json.loads(Path(style["settings_file"]).read_text()),
    }
    if style.get("displayed_unit_properties"):
        kw["displayed_unit_properties"] = style["displayed_unit_properties"]
    return kw


def main():
    parser = argparse.ArgumentParser(
        description="Launch the curation GUI for the chunk-tracked 48 h sort analyzer."
    )
    parser.add_argument(
        "--mode",
        choices=["web", "desktop"],
        default="web",
        help="'web' (Panel server, headless-friendly via SSH tunnel) or 'desktop' "
        "(Qt; needs a display, e.g. ssh -X / VNC). Default: web.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SIGUI_PORT", DEFAULT_PORT)),
        help=f"Port for web mode (default {DEFAULT_PORT} or $SIGUI_PORT).",
    )
    parser.add_argument(
        "--style",
        choices=sorted(STYLES),
        default=None,
        help="Named layout+settings preset (e.g. 'grahams_curation'). Default: GUI defaults.",
    )
    parser.add_argument(
        "--analyzer-path",
        default=str(ANALYZER_PATH),
        help="Analyzer to curate (default: the chunk-tracked analyzer_clustered.zarr). Use e.g. "
        ".../track_eval/mp_long_s2000_d170000/analyzer_tracks.zarr for the matching-pursuit 48 h tracks.",
    )
    args = parser.parse_args()
    # argparse has consumed our flags; clear sys.argv so Qt's QApplication (desktop
    # mode, via pyqtgraph mkQApp) doesn't reinterpret `--style` as its own built-in
    # widget-style option (Windows/Fusion) and warn.
    sys.argv = sys.argv[:1]

    if args.mode == "web":
        # Headless guard: never hand the URL to a terminal browser (w3m).
        os.environ.setdefault("BROWSER", "echo")

    import spikeinterface_gui as sg

    import spikeinterface as si

    # The GUI loads only the extensions each view needs, on demand, so skip the
    # eager full load (the waveforms extension alone is ~1.3 GB).
    sorting_analyzer = si.load_sorting_analyzer(args.analyzer_path, load_extensions=False)
    skw = style_kwargs(args.style)

    if args.mode == "web":
        print(
            f"\nServing curation GUI at  http://localhost:{args.port}/   (Ctrl-C to stop)"
        )
        print("From your laptop, open an SSH tunnel (use the SAME local/remote port):")
        print(f"    ssh -N -L {args.port}:localhost:{args.port} <you>@tononi-2")
        print(f"then browse to            http://localhost:{args.port}/\n", flush=True)
        sg.run_mainwindow(
            sorting_analyzer,
            mode="web",
            curation=True,
            address="localhost",
            port=args.port,
            panel_start_server_kwargs={
                "show": False
            },  # headless: do not auto-open a browser
            **skw,
        )
    else:
        print(
            "Launching desktop (Qt) curation GUI — requires a display (e.g. ssh -X / VNC).",
            flush=True,
        )
        sg.run_mainwindow(sorting_analyzer, mode="desktop", curation=True, **skw)


if __name__ == "__main__":
    main()

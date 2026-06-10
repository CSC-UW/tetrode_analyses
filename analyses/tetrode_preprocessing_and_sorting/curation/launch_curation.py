"""Launch the spikeinterface-gui curation GUI for the 12 h-block sort.

`--mode web` (default) runs the Panel web backend, headless-safe: it does NOT
auto-open a browser (tononi-2 has no display, and the default `www-browser` is
w3m, which would hijack the terminal). Reach it from your laptop over an SSH
tunnel (printed at startup). `--mode desktop` runs the Qt backend, which needs a
display (run over `ssh -X` or VNC).

Web port is pinned (--port / $SIGUI_PORT, default 8000) so the tunnel is
predictable; address stays "localhost" so a same-port tunnel is websocket-origin
valid. Web-only options are not passed in desktop mode. The GUI blocks until you
quit it.
"""
import argparse
import os
from pathlib import Path

ANALYZER_PATH = Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/"
    "sortings_seed42_pcafix/blosc-43200s-train3600s/analyzer.zarr"
)
DEFAULT_PORT = 8000


def main():
    parser = argparse.ArgumentParser(
        description="Launch the curation GUI for the 12 h-block sort analyzer."
    )
    parser.add_argument(
        "--mode", choices=["web", "desktop"], default="web",
        help="'web' (Panel server, headless-friendly via SSH tunnel) or 'desktop' "
             "(Qt; needs a display, e.g. ssh -X / VNC). Default: web.",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SIGUI_PORT", DEFAULT_PORT)),
        help=f"Port for web mode (default {DEFAULT_PORT} or $SIGUI_PORT).",
    )
    args = parser.parse_args()

    if args.mode == "web":
        # Headless guard: never hand the URL to a terminal browser (w3m).
        os.environ.setdefault("BROWSER", "echo")

    import spikeinterface as si
    import spikeinterface_gui as sg

    sorting_analyzer = si.load_sorting_analyzer(ANALYZER_PATH)

    if args.mode == "web":
        print(f"\nServing curation GUI at  http://localhost:{args.port}/   (Ctrl-C to stop)")
        print("From your laptop, open an SSH tunnel (use the SAME local/remote port):")
        print(f"    ssh -N -L {args.port}:localhost:{args.port} <you>@tononi-2")
        print(f"then browse to            http://localhost:{args.port}/\n", flush=True)
        sg.run_mainwindow(
            sorting_analyzer,
            mode="web",
            curation=True,
            address="localhost",
            port=args.port,
            panel_start_server_kwargs={"show": False},  # headless: do not auto-open a browser
        )
    else:
        print("Launching desktop (Qt) curation GUI — requires a display (e.g. ssh -X / VNC).",
              flush=True)
        sg.run_mainwindow(sorting_analyzer, mode="desktop", curation=True)


if __name__ == "__main__":
    main()

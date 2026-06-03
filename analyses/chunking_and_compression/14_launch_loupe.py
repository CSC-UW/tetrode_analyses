"""Launch a loupe viewer for the TTM-001 / TTM-NOD (2026-05-27_09-07-52) session.

Panes, top to bottom:
  1. Synthetic EMG — both estimators ("per_window" and "global") loaded from the
     precomputed ``synthetic_emg_methods.zarr`` (see ``analyses/emg/``).
  2. The 16 ~125 Hz sub-LFP tetrode-lead traces in a single dense subplot,
     colored by tetrode with the Open Ephys "Classic" palette.
  3. The MountainSort5 spike raster, split into one block per tetrode and colored
     by the same per-tetrode palette.

Spike times come from SpikeInterface directly: the source recording (which holds
the session-relative time vector) is registered to the sorting, and
``get_unit_spike_train(return_times=True)`` converts frames to seconds — so EMG,
LFP, and spikes share one clock.

    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/chunking_and_compression/14_launch_loupe.py
    uv run python ../tetrode_analyses/analyses/chunking_and_compression/14_launch_loupe.py --sorting wavpack-bps2.25
"""
import argparse
import pathlib
import re

import numpy as np
import polars as pl
import spikeinterface as si
import xarray as xr

import loupe as lp
from tetrode_analyses import experiment as exp
from tetrode_analyses.lfp import open_lfps_dataarray
from tetrode_analyses.viz import tetrode_color_map

SUBJECT, EXPERIMENT = "TTM-001", "TTM-NOD"
EMG_METHODS = ("per_window", "global")
EMG_COLORS = {"per_window": "#1f77b4", "global": "#ff7f0e"}


def build_spike_dataframe(aggregated, source_recording) -> pl.DataFrame:
    """Flatten an aggregated sorting into a (time, unit_id, tetrode) DataFrame.

    Registers ``source_recording`` (30 kHz, carrying the session-relative time
    vector) to the sorting so ``get_unit_spike_train(return_times=True)`` maps
    spike frames to session-relative seconds via SpikeInterface's own time
    handling — no manual frame arithmetic, and any inter-experiment gap in the
    time vector is honored.
    """
    aggregated.register_recording(source_recording)
    unit_ids = aggregated.get_unit_ids()
    groups = np.asarray(aggregated.get_property("group"))  # 0-based, per unit
    times_all, unit_all, tt_all = [], [], []
    for unit_id, group in zip(unit_ids, groups):
        times = aggregated.get_unit_spike_train(unit_id=unit_id, return_times=True)
        if times.size == 0:
            continue
        times_all.append(np.asarray(times))
        unit_all.append(np.full(times.size, unit_id))
        tt_all.append(np.full(times.size, f"TT{int(group) + 1}"))

    return pl.DataFrame(
        {
            "time": np.concatenate(times_all),
            "unit_id": np.concatenate(unit_all),
            "tetrode": np.concatenate(tt_all),
        }
    )


def load_emg_dataarray(emg_zarr: pathlib.Path) -> xr.DataArray:
    """Load the synthetic-EMG dataset as a (method, time) DataArray."""
    if not emg_zarr.exists():
        raise FileNotFoundError(
            f"synthetic EMG not found: {emg_zarr}\n"
            "Generate it first: cd gfys_workspace && uv run python "
            "../tetrode_analyses/analyses/emg/emg_methods_vs_swa.py --full"
        )
    ds = xr.open_zarr(emg_zarr)
    return ds[list(EMG_METHODS)].to_dataarray(dim="method").load()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sorting",
        choices=["blosc-zstd", "wavpack-bps2.25"],
        default="blosc-zstd",
        help="Which aggregated sorting to display (default: blosc-zstd).",
    )
    parser.add_argument(
        "--window-len", type=float, default=10.0, help="Initial window length (s)."
    )
    args = parser.parse_args()

    params = exp.load_experiment_params(
        exp.experiment_params_path(SUBJECT, EXPERIMENT)
    )
    root = pathlib.Path(params.openephys_session)
    sub_lfp = root / f"{root.name}.lfp.125hz.zarr"
    source_store = root / f"{root.name}.{args.sorting}.zarr"
    sorting_dir = root / "sortings" / args.sorting / "aggregated"
    emg_zarr = exp.experiment_dir(SUBJECT, EXPERIMENT) / "synthetic_emg_methods.zarr"

    # --- EMG: both estimators, top pane ---
    emg = load_emg_dataarray(emg_zarr)

    # --- LFP: 16 tetrode-lead traces, colored by tetrode ---
    da = open_lfps_dataarray(sub_lfp).load()
    tetrodes = [str(t) for t in np.asarray(da["tetrode"].values)]
    tt_num = np.array([int(re.findall(r"\d+", t)[0]) for t in tetrodes])
    da = da.assign_coords(tt_num=("channel", tt_num))  # stack traces in TT order
    palette = tetrode_color_map(tetrodes)

    # --- Spikes: raster split + colored by tetrode ---
    agg = si.load(str(sorting_dir))
    spikes = build_spike_dataframe(agg, si.read_zarr(str(source_store)))
    print(
        f"EMG {dict(emg.sizes)} | sub-LFP {da.shape} @ {float(da.attrs['fs']):.1f} Hz "
        f"({len(tetrodes)} leads) | {spikes.height} spikes from "
        f"{agg.get_num_units()} units ({args.sorting})"
    )

    lp.view(
        [
            lp.TraceConfig(
                emg,
                mode="stacked-subplots",
                hue="method",
                palette=EMG_COLORS,
                array_name="EMG",
            ),
            lp.TraceConfig(
                da,
                mode="dense",
                order_by="tt_num",
                descending=True,
                hue="tetrode",
                palette=palette,
                array_name="LFP",
            ),
            lp.RasterConfig(
                spikes,
                time_col="time",
                order_by="unit_id",
                split_by="tetrode",
                palette=palette,
                array_name="units",
            ),
        ],
        window_len=args.window_len,
    )


if __name__ == "__main__":
    main()

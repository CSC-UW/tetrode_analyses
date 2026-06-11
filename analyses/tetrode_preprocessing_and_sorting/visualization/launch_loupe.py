"""Launch a loupe viewer over a locally-downloaded TTM-001 / TTM-NOD viz bundle.

RUN THIS ON YOUR LOCAL MACHINE, pointing ``--data-dir`` at the folder that
``download_bundle.py`` pulled to your external drive::

    uv run python launch_loupe.py --data-dir /Volumes/MyDrive/ttm_nod
    uv run python launch_loupe.py --data-dir /Volumes/MyDrive/ttm_nod --window-len 30

The viewer is deliberately **SpikeInterface-free**: it reads only the plain
artifacts built by ``build_bundle.py`` (two xarray zarrs + a parquet), so the
install is just ``tetrode_analyses[viz]`` (base deps + ``loupe``) — no SI, no
ecephys, no source recording, no NFS.

Panes, top to bottom:
  1. Synthetic EMG — both estimators ("per_window" and "global").
  2. The 16 ~125 Hz sub-LFP tetrode-lead traces in one dense subplot, colored by
     tetrode with the Open Ephys "Classic" palette.
  3. The MountainSort5 spike raster in a single pane, colored by tetrode, with
     thin horizontal separators between tetrodes (loupe's ``horizontal_separators``).

Interval scoring is enabled via the bundle's ``state_definitions.json`` (the
sleep-scoring keymap + label colors from ``cnpix/sleepscore/launch_scoring``).

EMG, LFP, and spikes already share one session-relative clock — the LFP ``time``
coordinate and the parquet spike ``time`` were both written in session seconds by
``build_bundle.py``.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np
import polars as pl
import xarray as xr

import loupe as lp
from tetrode_analyses.viz import _tetrode_sort_key, tetrode_color_map

EMG_FILENAME = "synthetic_emg_methods.zarr"
LFP_FILENAME = "lfp.125hz.zarr"
SPIKES_FILENAME = "spikes.parquet"
STATE_DEFINITIONS_FILENAME = "state_definitions.json"  # loupe scoring keymap + colors
EMG_METHODS = ("per_window", "global")
EMG_COLORS = {"per_window": "#1f77b4", "global": "#ff7f0e"}
SEPARATOR_PARAMS = {"gap": 0.6, "color": "#888888", "width": 1.0}


def load_emg_dataarray(data_dir: pathlib.Path) -> xr.DataArray:
    """Load both synthetic-EMG estimators as a (method, time) DataArray."""
    ds = xr.open_zarr(data_dir / EMG_FILENAME)
    return ds[list(EMG_METHODS)].to_dataarray(dim="method").load()


def tetrode_separators(spikes: pl.DataFrame) -> tuple[list[str], list[int] | None]:
    """Ordered tetrode labels + unit_id boundaries for ``horizontal_separators``.

    Returns the unit_id below which to draw a line at each tetrode boundary (the
    last unit_id of each tetrode except the final one), in tetrode order — what
    loupe wants for a single-pane, unit_id-ordered raster. Returns ``None`` for the
    boundaries if unit_ids are not tetrode-contiguous (the caller then falls back
    to ``split_by="tetrode"`` rather than draw misplaced lines).
    """
    per_unit = spikes.select("unit_id", "tetrode").unique().sort("unit_id")
    tetrodes = sorted(per_unit["tetrode"].unique().to_list(), key=_tetrode_sort_key)
    spans = {
        tt: (
            int(per_unit.filter(pl.col("tetrode") == tt)["unit_id"].min()),
            int(per_unit.filter(pl.col("tetrode") == tt)["unit_id"].max()),
        )
        for tt in tetrodes
    }
    ordered = [spans[tt] for tt in tetrodes]
    contiguous = all(
        ordered[i][1] < ordered[i + 1][0] for i in range(len(ordered) - 1)
    )
    if not contiguous:
        return tetrodes, None
    boundaries = [spans[tt][1] for tt in tetrodes[:-1]]
    return tetrodes, boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Bundle folder downloaded by download_bundle.py.",
    )
    parser.add_argument(
        "--window-len", type=float, default=10.0, help="Initial window length (s)."
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    if not (data_dir / SPIKES_FILENAME).exists():
        raise FileNotFoundError(
            f"{SPIKES_FILENAME} not found in {data_dir}. Build the bundle on "
            "tononi-2 (build_bundle.py) and download it (download_bundle.py) first."
        )

    # --- EMG: both estimators, top pane ---
    emg = load_emg_dataarray(data_dir)

    # --- LFP: 16 tetrode-lead traces, colored by tetrode ---
    da = xr.open_zarr(data_dir / LFP_FILENAME)["lfp"].load()
    tetrodes = [str(t) for t in np.asarray(da["tetrode"].values)]
    tt_num = np.array([int(re.findall(r"\d+", t)[0]) for t in tetrodes])
    da = da.assign_coords(tt_num=("channel", tt_num))  # stack traces in TT order
    palette = tetrode_color_map(tetrodes)

    # --- Scoring state definitions (keymap + label colors), enables interval labeling ---
    state_path = data_dir / STATE_DEFINITIONS_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(
            f"{STATE_DEFINITIONS_FILENAME} not found in {data_dir}. Rebuild the "
            "bundle on tononi-2 (build_bundle.py) and re-download it."
        )

    # --- Spikes: single-pane raster, tetrode-colored, separators between tetrodes ---
    spikes = pl.read_parquet(data_dir / SPIKES_FILENAME)
    tt_order, separators = tetrode_separators(spikes)
    palette = tetrode_color_map(tt_order) | palette  # ensure every tetrode has a color
    print(
        f"EMG {dict(emg.sizes)} | sub-LFP {da.shape} @ {float(da.attrs['fs']):.1f} Hz "
        f"({len(tetrodes)} leads) | {spikes.height} spikes from "
        f"{spikes['unit_id'].n_unique()} units across {len(tt_order)} tetrodes"
    )
    if separators is None:
        print(
            "  NOTE: unit_ids not tetrode-contiguous; falling back to "
            "split_by='tetrode' (no horizontal separators)."
        )

    if separators is None:
        raster = lp.RasterConfig(
            spikes,
            time_col="time",
            order_by="unit_id",
            split_by="tetrode",
            palette=palette,
            array_name="units",
        )
    else:
        raster = lp.RasterConfig(
            spikes,
            time_col="time",
            order_by="unit_id",
            split_by=None,
            hue="tetrode",
            palette=palette,
            horizontal_separators=separators,
            separator_params=SEPARATOR_PARAMS,
            array_name="units",
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
            raster,
        ],
        window_len=args.window_len,
        state_definitions=str(state_path),
    )


if __name__ == "__main__":
    main()

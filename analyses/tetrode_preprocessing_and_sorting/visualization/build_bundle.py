"""Assemble a self-contained "viz bundle" for the local loupe viewer.

RUN THIS ON tononi-2 (the box that holds the session data), inside the workspace
environment::

    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/visualization/build_bundle.py

It gathers everything ``visualization/launch_loupe.py`` needs — and nothing it
doesn't — into one folder, so the viewer can run on a laptop with no
SpikeInterface, no source recording, and no NFS/NVME mounts. The companion
``download_bundle.py`` then rclones this folder to an external drive.

Bundle contents (default ``<session>/viz_bundle/``)::

    synthetic_emg_methods.zarr   # copied as-is (already a plain xarray zarr)
    lfp.125hz.zarr               # re-exported as a PLAIN xarray zarr (SI-free read)
    spikes.parquet               # one row per spike: time[s], unit_id, tetrode
    manifest.json                # provenance + per-tetrode separator boundaries

Why a prep step (rather than downloading the raw artifacts)?

  * The 125 Hz sub-LFP the viewer draws does not exist until ``make_subsampled_lfp``
    builds it from the 625 Hz LFP, and a SpikeInterface-saved zarr is not directly
    ``xr.open_zarr``-able — so we re-export it as a plain xarray store here.
  * Mapping spike frames to session-relative seconds needs the source store's time
    vector (buried in a multi-hundred-GB zarr). We do that conversion once, on the
    fast box, and emit a compact parquet — instead of shipping the source store.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import sys

import numpy as np
import polars as pl
import spikeinterface as si

from tetrode_analyses import experiment as exp
from tetrode_analyses.lfp import make_subsampled_lfp, open_lfps_dataarray
from tetrode_analyses.viz import _tetrode_sort_key

SUBJECT, EXPERIMENT = "TTM-001", "TTM-NOD"
EMG_FILENAME = "synthetic_emg_methods.zarr"
DEFAULT_SORTINGS_SUBDIR = "sortings_seed42_pcafix"
DEFAULT_SORTING = "blosc-43200s-train3600s"  # 48 h, 12 h blocks, 1 h training window
# Source store whose session-relative time vector maps spike frames -> seconds.
# Any compressor variant carries the same recording clock; blosc-zstd is lossless.
DEFAULT_SOURCE_COMPRESSOR = "blosc-zstd"


def build_spike_dataframe(aggregated, source_recording) -> pl.DataFrame:
    """Flatten an aggregated sorting into a (time, unit_id, tetrode) DataFrame.

    Maps every spike frame to session-relative seconds in a single vectorized pass:
    take the full spike vector (frames + unit indices, sorted by frame across all
    units) and convert with one ``source_recording.sample_index_to_time`` call,
    using SpikeInterface's native time API so the recording's session-relative time
    vector (and any inter-experiment gap) is honored — no manual frame arithmetic.

    One pass matters here: every unit fires across the full 48 h, so the per-unit
    ``get_unit_spike_train(return_times=True)`` path re-decompresses the entire
    (lazy, multi-GB) time vector once *per unit* — ~N_units passes. Converting the
    whole sorted spike vector at once decompresses it exactly once.
    """
    sv = aggregated.to_spike_vector()  # sorted by sample_index across all units
    frames = np.asarray(sv["sample_index"])
    unit_idx = np.asarray(sv["unit_index"])  # index into aggregated.unit_ids
    times = np.asarray(source_recording.sample_index_to_time(frames, segment_index=0))
    unit_ids = np.asarray(aggregated.unit_ids)
    groups = np.asarray(aggregated.get_property("group"))  # 0-based, per unit

    return pl.DataFrame(
        {
            "time": times.astype(np.float64),
            "unit_id": unit_ids[unit_idx].astype(np.int32),
            "tt_num": (groups[unit_idx] + 1).astype(np.int16),
        }
    ).with_columns(
        ("TT" + pl.col("tt_num").cast(pl.String)).alias("tetrode")
    ).drop("tt_num")


def tetrode_separator_boundaries(spikes: pl.DataFrame) -> tuple[list[str], list[int]]:
    """Return ordered tetrode labels and the unit_id below which to draw a line.

    Boundaries are the *last* unit_id of each tetrode (except the final one), in
    tetrode order — exactly the values ``loupe``'s ``horizontal_separators`` wants
    to delimit per-tetrode blocks of a single-pane, unit_id-ordered raster.

    Assumes unit_ids are tetrode-contiguous (true for ``aggregate_units``
    numbering: TT1's units precede TT2's, etc.). Verifies it and raises if not, so
    the viewer can fall back to ``split_by="tetrode"`` rather than draw wrong lines.
    """
    per_unit = (
        spikes.select("unit_id", "tetrode")
        .unique()
        .sort("unit_id")
    )
    tetrodes = sorted(per_unit["tetrode"].unique().to_list(), key=_tetrode_sort_key)
    boundaries: list[int] = []
    for tt in tetrodes[:-1]:
        last_uid = per_unit.filter(pl.col("tetrode") == tt)["unit_id"].max()
        boundaries.append(int(last_uid))

    # Contiguity check: each tetrode must occupy a single contiguous unit_id span.
    spans = {
        tt: (
            int(per_unit.filter(pl.col("tetrode") == tt)["unit_id"].min()),
            int(per_unit.filter(pl.col("tetrode") == tt)["unit_id"].max()),
        )
        for tt in tetrodes
    }
    ordered = [spans[tt] for tt in tetrodes]
    contiguous = all(lo <= hi for lo, hi in ordered) and all(
        ordered[i][1] < ordered[i + 1][0] for i in range(len(ordered) - 1)
    )
    if not contiguous:
        raise ValueError(
            "unit_ids are not tetrode-contiguous; horizontal separators would be "
            f"wrong. Per-tetrode (min,max) unit_id spans: {spans}"
        )
    return tetrodes, boundaries


def _copytree(src: pathlib.Path, dst: pathlib.Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            print(f"  exists, skipping: {dst}")
            return
        shutil.rmtree(dst)
    print(f"  copy {src} -> {dst}")
    shutil.copytree(src, dst)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # live progress when redirected to a log
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorting", default=DEFAULT_SORTING)
    parser.add_argument("--sortings-subdir", default=DEFAULT_SORTINGS_SUBDIR)
    parser.add_argument("--source-compressor", default=DEFAULT_SOURCE_COMPRESSOR)
    parser.add_argument(
        "--bundle",
        type=pathlib.Path,
        default=None,
        help="Output bundle dir (default: <session>/viz_bundle).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    params = exp.load_experiment_params(
        exp.experiment_params_path(SUBJECT, EXPERIMENT)
    )
    root = pathlib.Path(params.openephys_session)
    lfp_625 = pathlib.Path(params.lfp_zarr)
    sub_lfp = root / f"{root.name}.lfp.125hz.zarr"
    source_store = root / f"{root.name}.{args.source_compressor}.zarr"
    aggregated_dir = root / args.sortings_subdir / args.sorting / "aggregated"
    emg_src = exp.experiment_dir(SUBJECT, EXPERIMENT) / EMG_FILENAME

    bundle = args.bundle or (root / "viz_bundle")
    bundle.mkdir(parents=True, exist_ok=True)
    print(f"Bundle -> {bundle}")

    # --- EMG: copy the plain xarray zarr verbatim ---
    print("[1/4] EMG")
    _copytree(emg_src, bundle / EMG_FILENAME, overwrite=args.overwrite)

    # --- 125 Hz sub-LFP: ensure the SI store exists, then re-export plain xarray ---
    print("[2/4] 125 Hz LFP")
    if not sub_lfp.exists():
        print(f"  building 125 Hz sub-LFP (missing): {sub_lfp}")
        make_subsampled_lfp(lfp_625, sub_lfp, resample_rate=125, n_jobs=16)
    bundle_lfp = bundle / "lfp.125hz.zarr"
    if bundle_lfp.exists() and not args.overwrite:
        print(f"  exists, skipping: {bundle_lfp}")
        da = open_lfps_dataarray(sub_lfp)
    else:
        if bundle_lfp.exists():
            shutil.rmtree(bundle_lfp)
        da = open_lfps_dataarray(sub_lfp)
        print(f"  re-export {da.shape} @ {float(da.attrs['fs']):.1f} Hz -> {bundle_lfp}")
        da.to_dataset(name="lfp").to_zarr(bundle_lfp, mode="w")
    tetrodes_lfp = [str(t) for t in np.asarray(da["tetrode"].values)]

    # --- Spikes: frames -> session seconds via SI native time API, to parquet ---
    print("[3/4] spikes")
    agg = si.load(str(aggregated_dir))
    spikes = build_spike_dataframe(agg, si.read_zarr(str(source_store)))
    spikes_path = bundle / "spikes.parquet"
    spikes.write_parquet(spikes_path, compression="zstd")
    tetrode_order, separator_boundaries = tetrode_separator_boundaries(spikes)
    print(
        f"  {spikes.height} spikes, {agg.get_num_units()} units, "
        f"{len(tetrode_order)} tetrodes -> {spikes_path}"
    )

    # --- Manifest ---
    print("[4/4] manifest")
    manifest = {
        "subject": SUBJECT,
        "experiment": EXPERIMENT,
        "session": root.name,
        "sorting": args.sorting,
        "sortings_subdir": args.sortings_subdir,
        "source_compressor": args.source_compressor,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "lfp": {
            "fs": float(da.attrs["fs"]),
            "n_samples": int(da.sizes["time"]),
            "n_channels": int(da.sizes["channel"]),
            "tetrodes": tetrodes_lfp,
        },
        "spikes": {
            "n_spikes": int(spikes.height),
            "n_units": int(agg.get_num_units()),
            "tetrode_order": tetrode_order,
            # unit_id values below which to draw a per-tetrode separator line:
            "separator_boundaries": separator_boundaries,
        },
        "emg_methods": ["per_window", "global"],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nDONE. Download with:\n  python download_bundle.py --dest <external-drive>/ttm_nod")


if __name__ == "__main__":
    main()

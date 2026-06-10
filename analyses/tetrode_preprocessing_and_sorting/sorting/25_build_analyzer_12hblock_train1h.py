"""Build a SortingAnalyzer (Zarr) for the full-48 h, 12 h-block / 1 h-training sort.

Pairs the aggregated sorting at
``sortings_seed42_pcafix/blosc-43200s-train3600s/aggregated`` (232 units, 81.3 M
spikes; produced by ``24_sort_12h_blocks_1h_train.py``) with the SAME preprocessed
recording it was sorted on -- bandpass(300-6000 Hz) + global CMR, float32 -- and
computes the standard postprocessing + metrics extension suite into ``analyzer.zarr``.

Recording is left LAZY (not re-materialized): the Zarr analyzer stores the source
zarr path + the preprocessing chain, so it reloads self-contained without a ~1.3 TB
binary cache. The recording keeps its real session-relative time vector; the
installed SpikeInterface checkout loads it as a read-only memmap (commit 31f1c433c /
PR #4608), so it is not materialized into RAM under n_jobs multiprocessing -- no
``reset_times()`` needed.

Sparsity is tetrode-direct: ``ChannelSparsity.from_property(by_property="group")``
maps each unit to the 4 channels of its own tetrode group (more direct than a radius
heuristic; both the recording and the sorting carry the ``group`` property).

NB: all work runs under ``if __name__ == "__main__":``. SpikeInterface's
``principal_components`` fit uses a ``ProcessPoolExecutor`` and Python 3.14 defaults
the multiprocessing start method to ``forkserver``, so worker processes RE-IMPORT
this module -- without the guard they would re-run the build and clobber the Zarr.
"""
import json
import shutil
import time
import pathlib
import spikeinterface as si
from spikeinterface.core.sparsity import ChannelSparsity
from tetrode_analyses.sorting import preprocess_for_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
SORT_DIR = SR / "blosc-43200s-train3600s"
AGG = SORT_DIR / "aggregated"
OUT = SORT_DIR / "analyzer.zarr"
DISK_FLOOR_GB = 50  # analyzer.zarr is a few GB (per-spike amps/locs); no big materialize

# All requested extensions, in a single dependency-ordered compute. Only correlograms
# gets explicit params (window_ms=100 / bin_ms=5 -> 40 lag bins); the rest use SI
# defaults (random_spikes 500/unit, waveforms 1.0/2.0 ms, ...). quality_metrics uses
# the default metric set, which includes PCA-based metrics since principal_components
# is present.
EXTENSIONS = {
    "random_spikes": {},
    "noise_levels": {},
    "waveforms": {},
    "templates": {},
    "unit_locations": {},
    "spike_locations": {},
    "spike_amplitudes": {},
    "isi_histograms": {},
    "correlograms": {"window_ms": 100.0, "bin_ms": 5.0},
    "principal_components": {},
    "template_similarity": {},
    "quality_metrics": {},
    "template_metrics": {},
}


def main():
    job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")
    si.set_global_job_kwargs(**job_kwargs)

    fg = shutil.disk_usage("/nvme").free / 1e9
    print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB})", flush=True)
    if fg < DISK_FLOOR_GB:
        raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")

    # ---- load recording (lazy bandpass + global CMR) + sorting ----
    print(f"[{time.strftime('%T')}] loading recording + sorting ...", flush=True)
    rec = si.read_zarr(str(BLOSC))
    pp = preprocess_for_sorting(rec, cmr="global")
    sorting = si.load(str(AGG))
    print(f"  {sorting.get_num_units()} units | {pp.get_num_channels()} ch | "
          f"{pp.get_num_frames()/30000/3600:.1f} h | scaleable={pp.has_scaleable_traces()}", flush=True)

    # ---- tetrode-direct sparsity: each unit -> the 4 channels of its own group ----
    sparsity = ChannelSparsity.from_property(sorting, pp, by_property="group")
    chans_per_unit = sparsity.mask.sum(axis=1)
    print(f"[{time.strftime('%T')}] sparsity by group: {int(chans_per_unit.min())}-"
          f"{int(chans_per_unit.max())} channels/unit", flush=True)

    # ---- create Zarr analyzer + compute extensions ----
    print(f"[{time.strftime('%T')}] creating analyzer (zarr, return_in_uV) -> {OUT} ...", flush=True)
    analyzer = si.create_sorting_analyzer(
        sorting, pp, format="zarr", folder=str(OUT),
        sparsity=sparsity, return_in_uV=True, overwrite=True,
    )  # explicit sparsity -> `sparse` flag is ignored

    print(f"[{time.strftime('%T')}] computing {len(EXTENSIONS)} extensions ...", flush=True)
    t0 = time.perf_counter()
    analyzer.compute(EXTENSIONS)  # dependency-ordered; uses global job_kwargs
    compute_min = (time.perf_counter() - t0) / 60
    present = sorted(analyzer.get_saved_extension_names())
    print(f"[{time.strftime('%T')}] computed in {compute_min:.1f} min | extensions: {present}", flush=True)

    # ---- summary ----
    qm_df = analyzer.get_extension("quality_metrics").get_data()
    tm_df = analyzer.get_extension("template_metrics").get_data()
    summary = {
        "analyzer": str(OUT),
        "source_sorting": str(AGG),
        "n_units": int(sorting.get_num_units()),
        "n_samples": int(pp.get_num_frames()),
        "duration_h": round(pp.get_num_frames() / 30000 / 3600, 2),
        "format": "zarr",
        "return_in_uV": True,
        "sparsity": {"method": "by_property:group",
                     "channels_per_unit": [int(chans_per_unit.min()), int(chans_per_unit.max())]},
        "job_kwargs": {"n_jobs": -1, "chunk_duration": "1s"},
        "compute_min": round(compute_min, 1),
        "extensions": present,
        "correlograms_params": EXTENSIONS["correlograms"],
        "quality_metrics_columns": list(qm_df.columns),
        "quality_metrics_rows": int(len(qm_df)),
        "template_metrics_columns": list(tm_df.columns),
        "template_metrics_rows": int(len(tm_df)),
    }
    (SR / "analyzer_12hblock_train1h_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- 48 h SortingAnalyzer @ 12 h blocks / 1 h training ---", flush=True)
    print(f"analyzer    : {OUT}", flush=True)
    print(f"units       : {summary['n_units']}", flush=True)
    print(f"compute     : {compute_min:.1f} min", flush=True)
    print(f"extensions  : {len(present)} -> {present}", flush=True)
    print("RESULT " + json.dumps({k: summary[k] for k in
          ("n_units", "duration_h", "compute_min", "extensions", "correlograms_params")}), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

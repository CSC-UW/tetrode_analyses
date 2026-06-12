"""Build a SortingAnalyzer (Zarr) + quality metrics for the chunk-tracked 48 h sort.

This is the curation/QC deliverable for the tracking output: it pairs the assembled
global sorting (``tracked_48h/global_sorting_clustered.npz``, produced by
``36_track_global_cluster.py`` -- gap-tolerant global clustering of the per-chunk MS5
scheme-2 sorts) with the SAME preprocessed recording the chunks were sorted on
(bandpass 300-6000 Hz + global CMR, float32) and computes the standard postprocessing
+ quality-metrics suite. Counterpart to ``28_build_analyzer_singleblock_scheme2.py``
(the failed 48 h single-shot scheme-2 sort) so the two analyzers are directly
comparable, EXCEPT for the geometry-dependent extensions (see below).

The global CMR is computed per frame, so the per-chunk crops the units were sorted on
carry byte-identical traces to the full-recording preprocessing at the same absolute
frames; the assembled sorting is in absolute frames, so this full-48 h preprocessed
recording IS the exact recording that produced the sort (cf. the "exact recording for
a sort" rule).

GEOMETRY-FREE METRIC SET: the tetrode geometry is fictional (4 co-located wires, no
known coordinates), so the positional extensions are dropped -- NO ``unit_locations``,
NO ``spike_locations``, and the ``drift`` quality metric (which needs spike_locations)
is excluded. Everything kept is geometry-free: spike-train statistics, waveform/SNR/
amplitude metrics, and PCA-based isolation metrics (mahalanobis / d_prime /
nearest_neighbor / silhouette operate on PCA of the sparse 4-channel waveforms, no
channel coordinates). Sparsity is tetrode-direct
(``ChannelSparsity.from_property(by_property="group")``).

Recording is left LAZY (not re-materialized): the Zarr analyzer stores the source zarr
path + preprocessing chain, reloading self-contained without a ~1.3 TB binary cache.

NB: all work runs under ``if __name__ == "__main__":`` -- SI's principal_components
fit uses a ProcessPoolExecutor and Python 3.14 defaults to forkserver, so workers
re-import this module; without the guard they would re-run the build and clobber the
Zarr.
"""
import json
import shutil
import time
import pathlib

import numpy as np
import spikeinterface as si
from spikeinterface.core.sparsity import ChannelSparsity

from tetrode_analyses.sorting import preprocess_for_sorting

from _track_eval import TIERS, isolation_tier_mask

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
TRACKED = SR / "tracked_48h"
NPZ = TRACKED / "global_sorting_clustered.npz"
OUT = TRACKED / "analyzer_clustered.zarr"
FS = 30000.0
DISK_FLOOR_GB = 50  # analyzer.zarr is a few GB (per-spike amps + PCA); no big materialize
N_JOBS = 64  # polite: 224 cores, leave headroom for a co-running GPU sort's host threads

# Geometry-free quality metrics (validated names from 26_profile_quality_metrics.py),
# MINUS `drift` (needs spike_locations -> fictional geometry). The four PCA metrics
# (mahalanobis/d_prime/nearest_neighbor/silhouette) need principal_components.
QM_METRICS = [
    "num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation",
    "rp_violation", "sliding_rp_violation", "synchrony", "firing_range",
    "amplitude_cv", "amplitude_cutoff", "noise_cutoff", "amplitude_median",
    "sd_ratio", "mahalanobis", "d_prime", "nearest_neighbor", "silhouette",
]

# Geometry-free extension suite (NO unit_locations / spike_locations).
EXTENSIONS = {
    "random_spikes": {},
    "noise_levels": {},
    "waveforms": {},
    "templates": {},
    "spike_amplitudes": {},
    "isi_histograms": {},
    "correlograms": {"window_ms": 100.0, "bin_ms": 5.0},
    "principal_components": {},
    "template_similarity": {},
    "template_metrics": {},
    "quality_metrics": {"metric_names": QM_METRICS},
}


def load_npz_sorting(npz_path):
    """Reconstruct the assembled global sorting (absolute frames) from its .npz."""
    d = np.load(npz_path)
    uids = d["unit_ids"]
    units = {int(u): d[f"st_{u}"].astype(np.int64) for u in uids}
    srt = si.NumpySorting.from_unit_dict([units], sampling_frequency=FS)
    srt.set_property("group", np.asarray(d["group"]))
    return srt


def main():
    job_kwargs = dict(n_jobs=N_JOBS, progress_bar=True, chunk_duration="1s")
    si.set_global_job_kwargs(**job_kwargs)

    fg = shutil.disk_usage("/nvme").free / 1e9
    print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB})", flush=True)
    if fg < DISK_FLOOR_GB:
        raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")
    if not NPZ.exists():
        raise SystemExit(f"ABORT: clustered sorting not found at {NPZ}")

    # ---- recording (lazy bandpass + global CMR) + assembled sorting ----
    print(f"[{time.strftime('%T')}] loading recording + clustered sorting ...", flush=True)
    rec = si.read_zarr(str(BLOSC))
    pp = preprocess_for_sorting(rec, cmr="global")
    if pp.get_property("group") is None:
        raise SystemExit("ABORT: recording has no 'group' property (probegroup not attached)")
    sorting = load_npz_sorting(NPZ)
    print(f"  {sorting.get_num_units()} units | {pp.get_num_channels()} ch | "
          f"{pp.get_num_frames() / FS / 3600:.1f} h | scaleable={pp.has_scaleable_traces()}", flush=True)

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
    )

    print(f"[{time.strftime('%T')}] computing {len(EXTENSIONS)} extensions (geometry-free) ...", flush=True)
    t0 = time.perf_counter()
    analyzer.compute(EXTENSIONS)
    compute_min = (time.perf_counter() - t0) / 60
    present = sorted(analyzer.get_saved_extension_names())
    print(f"[{time.strftime('%T')}] computed in {compute_min:.1f} min | extensions: {present}", flush=True)

    # ---- summary + a quick curation-tier headcount (geometry-free isolation) ----
    qm = analyzer.get_extension("quality_metrics").get_data()
    tm = analyzer.get_extension("template_metrics").get_data()

    # Canonical isolation tiers (rp_contamination OR sliding_rp_violation + FR floor),
    # thresholds from ecephys.wne.siutils via _track_eval (SPOT).
    tier_masks = {t: isolation_tier_mask(qm, t) for t in TIERS}
    tier_counts = {t: int(m.sum()) for t, m in tier_masks.items()}
    tier_frac = {t: round(float(np.mean(m)), 3) for t, m in tier_masks.items()}

    summary = {
        "analyzer": str(OUT),
        "source_sorting": str(NPZ),
        "provenance": "chunk-tracked MS5 scheme2 (36_track_global_cluster.py, global_sorting_clustered)",
        "n_units": int(sorting.get_num_units()),
        "n_samples": int(pp.get_num_frames()),
        "duration_h": round(pp.get_num_frames() / FS / 3600, 2),
        "format": "zarr",
        "return_in_uV": True,
        "geometry_free": True,
        "dropped_extensions": ["unit_locations", "spike_locations"],
        "dropped_quality_metrics": ["drift"],
        "sparsity": {"method": "by_property:group",
                     "channels_per_unit": [int(chans_per_unit.min()), int(chans_per_unit.max())]},
        "job_kwargs": {"n_jobs": N_JOBS, "chunk_duration": "1s"},
        "compute_min": round(compute_min, 1),
        "extensions": present,
        "quality_metrics_columns": list(qm.columns),
        "quality_metrics_rows": int(len(qm)),
        "template_metrics_columns": list(tm.columns),
        "curation_tier_counts": tier_counts,
        "curation_tier_frac": tier_frac,
    }
    (SR / "analyzer_tracked_clustered_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- 48 h SortingAnalyzer @ chunk-tracked (clustered) sort ---", flush=True)
    print(f"analyzer    : {OUT}", flush=True)
    print(f"units       : {summary['n_units']}  | compute {compute_min:.1f} min", flush=True)
    print(f"extensions  : {len(present)} (geometry-free) -> {present}", flush=True)
    print(f"curation    : {tier_counts}  (frac {tier_frac})", flush=True)
    print("RESULT " + json.dumps({k: summary[k] for k in
          ("n_units", "duration_h", "compute_min", "curation_tier_counts")}), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

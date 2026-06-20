"""Build a loadable curation SortingAnalyzer on the PRODUCTION deliverable ``assembled_prod``.

``assembled_prod`` (script 97) = 71 SUA (is_mua=False) + 16 per-tetrode MUA pseudo-units (is_mua=True),
91.7M spikes. This wraps it with the EXACT recording that produced the sort (OUT/binary, the materialized
bandpass+CMR span -- ``materialize_span`` returns ``si.load(OUT/'binary')``) into a geometry-free,
group-sparse SortingAnalyzer, mirroring 56_build_carry_forward_analyzer.py so it opens in
spikeinterface-gui / phy exactly like analyzer_clustered.zarr. Geometry-free: NO unit/spike locations or
drift (tetrode geometry is fictional); per-tetrode (group) sparsity. The ``is_mua`` unit property carries
through the analyzer, so MUA pseudo-units stay distinguishable during curation.

The one full-recording pass is ``spike_amplitudes`` (all 91.7M spikes, ~19M of them in the pooled MUA
units); ``--light`` skips it + the amplitude/PCA-based quality metrics for a fast SUA-shape-only build.
Everything else is subsampled (waveforms/PCA on <=500 spikes/unit) or binned (correlograms/ISI).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/98_build_analyzer_prod.py
    # fast shape-only build (no per-spike amplitudes / PCA-metrics):
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/98_build_analyzer_prod.py --light
"""
import argparse
import pathlib

import spikeinterface as si
from spikeinterface.core import ChannelSparsity

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sorting", default="assembled_prod", help="assembled-sorting folder under the run dir")
    ap.add_argument("--out-name", default="analyzer_prod.zarr")
    ap.add_argument("--light", action="store_true",
                    help="skip spike_amplitudes + amplitude/PCA-based quality metrics (fast, no full pass)")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    analyzer_path = OUT / args.out_name

    rec = si.load(OUT / "binary")                       # the exact recording that produced the sort
    sorting = si.load(OUT / args.sorting)
    has_mua = "is_mua" in sorting.get_property_keys()
    n_mua = int(sum(bool(x) for x in sorting.get_property("is_mua"))) if has_mua else 0
    print(f"recording {rec.get_duration() / 3600:.1f}h {rec.get_num_channels()}ch | "
          f"sorting {sorting.get_num_units()} units ({sorting.get_num_units() - n_mua} SUA + {n_mua} MUA)",
          flush=True)

    sparsity = ChannelSparsity.from_property(sorting, rec, by_property="group")  # per-tetrode
    analyzer = si.create_sorting_analyzer(
        sorting, rec, format="zarr", folder=str(analyzer_path), overwrite=True,
        sparsity=sparsity, return_in_uV=True)
    # geometry-free curation set (mirrors analyzer_clustered.zarr; NO locations/drift)
    ext = {
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": 500},
        "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
        "templates": {"operators": ["average", "std"]},
        "noise_levels": {},
        "correlograms": {},
        "isi_histograms": {},
        "template_similarity": {},
        "principal_components": {"n_components": 5, "mode": "by_channel_local"},
    }
    if not args.light:
        ext["spike_amplitudes"] = {}                    # the only all-spikes (full-recording) pass
    analyzer.compute(ext, n_jobs=args.n_jobs, progress_bar=True)

    # SI metrics reorganized: isolation_distance + l_ratio -> "mahalanobis", nn_* -> "nearest_neighbor"
    # (old names raise a rename ValueError). Reproduces analyzer_clustered.zarr's QM column set; in --light
    # mode drop the metrics that need spike_amplitudes (amplitude_cutoff/_median).
    qm = ["firing_rate", "presence_ratio", "snr", "isi_violation", "rp_violation",
          "sliding_rp_violation", "d_prime", "mahalanobis", "nearest_neighbor", "silhouette"]
    if not args.light:
        qm += ["amplitude_cutoff", "amplitude_median"]
    analyzer.compute("quality_metrics", metric_names=qm)
    print(f"\nbuilt {analyzer_path}", flush=True)
    print("extensions:", sorted(analyzer.get_loaded_extension_names()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

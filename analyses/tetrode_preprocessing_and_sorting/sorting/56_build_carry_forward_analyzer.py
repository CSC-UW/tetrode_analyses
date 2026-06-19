"""Build a loadable curation SortingAnalyzer on the 48h carry-forward tracks.

The 48h deliverable (script 54) saved only the assembled NumpySorting; this wraps it with the
recording into a geometry-free, group-sparse SortingAnalyzer (curation extensions) so it can be opened
in spikeinterface-gui / phy exactly like the chunk-tracked analyzer_clustered.zarr. Geometry-free:
NO unit_locations/spike_locations/drift (tetrode geometry is fictional), per-tetrode (group) sparsity.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/56_build_carry_forward_analyzer.py
"""
import argparse
import pathlib

import spikeinterface as si
from spikeinterface.core import ChannelSparsity

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_long_s2000_d170000")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="suffix selecting a variant run (e.g. _dedup09); "
                    "reads assembled_reestimate<tag>, writes analyzer_tracks<tag>.zarr")
    ap.add_argument("--assembled", default=None, help="explicit assembled-sorting folder name under the run "
                    "dir (overrides assembled_reestimate<tag>; e.g. assembled_reseed_rs)")
    args = ap.parse_args()
    tag = args.tag
    analyzer_path = OUT / f"analyzer_tracks{tag}.zarr"
    rec = si.load(OUT / "binary")
    sorting = si.load(OUT / (args.assembled or f"assembled_reestimate{tag}"))
    print(f"recording {rec.get_duration()/3600:.1f}h {rec.get_num_channels()}ch | "
          f"sorting {sorting.get_num_units()} tracked units", flush=True)

    sparsity = ChannelSparsity.from_property(sorting, rec, by_property="group")  # per-tetrode
    analyzer = si.create_sorting_analyzer(
        sorting, rec, format="zarr", folder=str(analyzer_path), overwrite=True,
        sparsity=sparsity, return_in_uV=True)
    # geometry-free curation set (mirrors analyzer_clustered.zarr; NO locations/drift)
    analyzer.compute({
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": 500},
        "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
        "templates": {"operators": ["average", "std"]},
        "noise_levels": {},
        "spike_amplitudes": {},
        "correlograms": {},
        "isi_histograms": {},
        "template_similarity": {},
        "principal_components": {"n_components": 5, "mode": "by_channel_local"},
    }, n_jobs=16, progress_bar=True)
    # SI metrics reorganized: isolation_distance + l_ratio now come from the "mahalanobis"
    # metric, and nn_hit_rate/nn_miss_rate from "nearest_neighbor" (the old names raise a
    # rename ValueError). This list reproduces analyzer_clustered.zarr's QM column set.
    analyzer.compute("quality_metrics", metric_names=[
        "firing_rate", "presence_ratio", "snr", "isi_violation", "rp_violation",
        "sliding_rp_violation", "amplitude_cutoff", "amplitude_median", "d_prime",
        "mahalanobis", "nearest_neighbor", "silhouette"])
    print(f"\nbuilt {analyzer_path}", flush=True)
    print("extensions:", sorted(analyzer.get_loaded_extension_names()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

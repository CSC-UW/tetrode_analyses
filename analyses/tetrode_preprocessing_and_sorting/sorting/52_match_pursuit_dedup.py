"""PoC stage 0b: does a DEDUPLICATED template bank lift circus-omp fidelity?

Stage 0 (script 51) found matching pursuit detects 96% of well-isolated reference spikes, but
one-to-one unit agreement caps ~0.36 -- because MS5 oversplit produces near-identical template twins
and circus-omp splits each spike's assignment across them. This merges within-tetrode near-identical
units (template cosine >= threshold) before matching, and re-measures agreement vs the DEDUPED
reference. Expectation: agreement climbs toward the ~0.96 detection-recall ceiling.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/52_match_pursuit_dedup.py \
        [--start-s 36000] [--dur-s 180] [--n-jobs 16]
"""
import argparse
import pathlib

import numpy as np
import spikeinterface.comparison as sc

from _mp_common import build_templates_object, dedup_sorting, detection_recall, prepare_span, run_matching

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_poc")
AMP = {"amplitudes": [0.8, float("inf")]}  # raw-units sweet spot from stage 0


def evaluate(name, sorting, rec, n_jobs):
    templates, az = build_templates_object(sorting, rec, with_snr=True, n_jobs=n_jobs)
    mp, spikes = run_matching(rec, templates, method_kwargs=AMP, n_jobs=n_jobs)
    cmp = sc.compare_two_sorters(sorting, mp, sorting1_name="ref", sorting2_name="mp")
    agree = cmp.agreement_scores.values
    best = np.nanmax(agree, axis=1) if agree.size else np.zeros(sorting.get_num_units())
    qm = az.get_extension("quality_metrics").get_data()
    snr = qm["snr"].to_numpy()
    nsp = np.array([len(sorting.get_unit_spike_train(u)) for u in sorting.unit_ids])
    well = (snr >= 5.0) & (nsp >= 50)
    recall = detection_recall(sorting, spikes["sample_index"], well)
    n_ref = int(nsp.sum())
    wb = best[well] if well.any() else np.array([np.nan])
    print(f"{name:14s} units={sorting.get_num_units():4d} well={int(well.sum()):3d} | "
          f"well_med={np.median(wb):.3f} >=0.5={np.mean(wb>=0.5):.2f} >=0.8={np.mean(wb>=0.8):.2f} | "
          f"recall={recall:.3f} | mp/ref={len(spikes)/max(n_ref,1):.2f}x", flush=True)
    return np.median(wb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=36000.0)
    ap.add_argument("--dur-s", type=float, default=180.0)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    rec, ref = prepare_span(OUT, args.start_s, args.dur_s)
    print(f"span [{args.start_s:.0f},{args.start_s+args.dur_s:.0f})s  ref units={ref.get_num_units()}\n"
          f"{'setting':14s} {'':21s} fidelity (amp>=0.8)", flush=True)

    evaluate("nodedup", ref, rec, args.n_jobs)
    for cos in (0.95, 0.9, 0.85):
        merged = dedup_sorting(ref, rec, cosine_min=cos)
        evaluate(f"dedup{cos}", merged, rec, args.n_jobs)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Characterize the chunk-tracked clustered sort: do the WELL-ISOLATED units span long?

The tracking's whole purpose is long single-unit tracks. The QC analyzer
(37_build_analyzer_tracked.py) tells us how many units are well-isolated; this script
crosses that with each global unit's chunk-SPAN (from provenance_clustered.json) to answer
the question that actually decides whether the ~7 h-tracking output is useful: of the
well-isolated units, how long do they track, and how many per tetrode?

Reuses the canonical isolation tiers from _track_eval (rp_contamination OR
sliding_rp_violation + firing-rate floor; thresholds from ecephys.wne.siutils). Span = number
of distinct member chunks; approximate duration = (span+1)*stride for chunk_s=3600,
overlap=0.5 (stride 1800 s). No heavy compute -- loads the existing analyzer's
quality_metrics + the provenance JSON.
"""
import json
import pathlib

import numpy as np
import spikeinterface as si

from _track_eval import TIERS, isolation_tier_mask

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
TRACKED = SR / "tracked_48h"
ANALYZER = TRACKED / "analyzer_clustered.zarr"
PROV = TRACKED / "provenance_clustered.json"
OUTDIR = pathlib.Path(__file__).resolve().parent
N_CHUNKS = 96
STRIDE_S = 1800.0  # chunk_s=3600, overlap=0.5


def span_hours(span):
    return (span + 1) * STRIDE_S / 3600.0


def main():
    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    qm = analyzer.get_extension("quality_metrics").get_data()
    groups = np.asarray(analyzer.sorting.get_property("group"))
    uid_to_group = {int(u): int(g) for u, g in zip(analyzer.sorting.unit_ids, groups)}

    prov = json.loads(PROV.read_text())
    span = {int(gid): len({m[1] for m in members}) for gid, members in prov.items()}

    report = {"n_units": int(len(qm)), "n_chunks": N_CHUNKS, "tiers": {}}
    all_span = np.array([span.get(int(u), 0) for u in qm.index])
    report["all_units_span"] = {
        "median": float(np.median(all_span)),
        "p90": float(np.percentile(all_span, 90)),
        "max": int(all_span.max()),
        "frac_ge_half": round(float(np.mean(all_span >= N_CHUNKS / 2)), 3),
    }

    print(f"=== TRACKED CLUSTERED SORT: isolation x span ({len(qm)} units, {N_CHUNKS} chunks) ===", flush=True)
    print(f"{'tier':<14}{'n':>6}{'span_med':>10}{'span_p90':>10}{'span_max':>10}"
          f"{'~h_med':>8}{'~h_max':>8}{'>=8chunks':>11}", flush=True)
    for name in TIERS:
        mask = isolation_tier_mask(qm, name)
        ids = qm.index[mask]
        sp = np.array([span.get(int(u), 0) for u in ids])
        if len(sp) == 0:
            continue
        per_tet = {}
        for u in ids:
            g = uid_to_group[int(u)]
            per_tet[g] = per_tet.get(g, 0) + 1
        rec = {
            "n": int(len(sp)),
            "span_median": float(np.median(sp)),
            "span_p90": float(np.percentile(sp, 90)),
            "span_max": int(sp.max()),
            "approx_h_median": round(span_hours(np.median(sp)), 1),
            "approx_h_max": round(span_hours(sp.max()), 1),
            "n_span_ge_8": int(np.sum(sp >= 8)),       # ~>=7 h
            "n_span_ge_half": int(np.sum(sp >= N_CHUNKS / 2)),
            "tetrodes_with_good_units": len(per_tet),
            "per_tetrode_counts": {int(k): int(v) for k, v in sorted(per_tet.items())},
        }
        report["tiers"][name] = rec
        print(f"{name:<14}{rec['n']:>6}{rec['span_median']:>10.0f}{rec['span_p90']:>10.0f}"
              f"{rec['span_max']:>10}{rec['approx_h_median']:>8.1f}{rec['approx_h_max']:>8.1f}"
              f"{rec['n_span_ge_8']:>11}", flush=True)

    (TRACKED / "tracked_qc_characterization.json").write_text(json.dumps(report, indent=2))
    (OUTDIR / "track_qc_characterization.json").write_text(json.dumps(report, indent=2))
    print("\nRESULT " + json.dumps({"all_units_span": report["all_units_span"],
                                    "conservative": report["tiers"].get("conservative")}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

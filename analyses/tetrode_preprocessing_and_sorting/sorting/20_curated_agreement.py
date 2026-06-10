"""Curated lossless-vs-perturbed agreement on well-isolated units.

Defines "well-isolated" on the lossless reference (blosc-A) using the metrics that
actually discriminate on this 48h scheme-3 data (see 19_metric_distributions.py):
ISI violations + RP contamination + a firing-rate floor. presence_ratio is dropped
(it measures scheme-3 cross-block merging, not isolation; median 0.04) and
snr/amplitude_cutoff are dropped (non-discriminating: all units loud & complete).

For each perturbation (bps2.25, int16, bps6.0), compute the full match against the
lossless reference once, then restrict to each tier's good reference units and
report: fraction reproduced (matched ≥0.5) and mean agreement among matched. This
answers "of the well-isolated lossless units, are they preserved under compression?"
"""
import json
import pathlib
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
HERE = pathlib.Path(__file__).resolve().parent
METRICS_CSV = HERE / "metric_distributions_blosc-A.csv"

# confirmed "well-isolated" definition (ISI + RP + firing-rate; three tiers)
TIER_THRESH = {  # (isi_violations_ratio <, rp_contamination <, firing_rate >=)
    "permissive":   (0.5, 0.5, 0.2),
    "moderate":     (0.3, 0.3, 0.5),
    "conservative": (0.1, 0.1, 0.5),
}
PERTURBATIONS = {
    "wavpack-bps2.25": SR / "wavpack-bps2.25" / "aggregated",
    "blosc-int16":     SR / "blosc-int16" / "aggregated",
    "wavpack-bps6.0":  SR / "wavpack-bps6.0" / "aggregated",
}

df = pd.read_csv(METRICS_CSV, index_col=0)
ref = si.load(str(SR / "blosc-A" / "aggregated"))
ref_ids = np.asarray(ref.unit_ids)

def good_ids(tier):
    isi_t, rp_t, fr_t = TIER_THRESH[tier]
    ok = (df["isi_violations_ratio"] < isi_t) & (df["rp_contamination"] < rp_t) & (df["firing_rate"] >= fr_t)
    ok &= np.isfinite(df["isi_violations_ratio"]) & np.isfinite(df["rp_contamination"]) & np.isfinite(df["firing_rate"])
    ids = df.index[ok].to_numpy()
    return np.array([i for i in ids if i in set(ref_ids.tolist())])

good = {t: good_ids(t) for t in TIER_THRESH}
print("good-unit counts:", {t: int(len(v)) for t, v in good.items()}, flush=True)

out = {"definition": "ISI<t & RP<t & FR>=t (presence_ratio dropped); curate-on-lossless reference",
       "tier_thresholds": {t: {"isi<": v[0], "rp<": v[1], "fr>=": v[2]} for t, v in TIER_THRESH.items()},
       "n_good_units": {t: int(len(v)) for t, v in good.items()},
       "good_unit_ids": {t: [int(x) for x in v] for t, v in good.items()},
       "results": {}}

for pname, path in PERTURBATIONS.items():
    pert = si.load(str(path))
    cmp = sc.compare_two_sorters(ref, pert, sorting1_name="ref", sorting2_name=pname, match_score=0.5)
    m = cmp.get_matching()[0]          # best match per reference unit (-1 if none)
    ag = cmp.agreement_scores
    per_tier = {}
    # raw (all reference units) for context
    for label, ids in [("all_units", ref_ids), *[(t, good[t]) for t in TIER_THRESH]]:
        ids = np.asarray(ids)
        matched = [u for u in ids if m.get(u, -1) != -1]
        frac = len(matched) / len(ids) if len(ids) else float("nan")
        mean_ag = float(np.nanmean([ag.loc[u, m[u]] for u in matched])) if matched else float("nan")
        per_tier[label] = {"n_ref": int(len(ids)), "matched": len(matched),
                           "match_frac": round(frac, 3), "mean_agreement": round(mean_ag, 4)}
    out["results"][pname] = per_tier
    print(f"\n{pname}:", flush=True)
    for label, r in per_tier.items():
        print(f"  {label:<13} n_ref={r['n_ref']:>3} matched={r['matched']:>3} "
              f"frac={r['match_frac']:.3f} mean_ag={r['mean_agreement']}", flush=True)

(SR / "comparison_curated_summary.json").write_text(json.dumps(out, indent=2))

# ready-to-read table
print("\n--- CURATED AGREEMENT (mean agreement among matched / match fraction) ---", flush=True)
hdr = f"{'tier':<13}{'n_good':>7}" + "".join(f"{p:>22}" for p in PERTURBATIONS)
print(hdr, flush=True)
for label in ["all_units", *TIER_THRESH]:
    ng = out["results"]["wavpack-bps2.25"][label]["n_ref"]
    cells = []
    for p in PERTURBATIONS:
        r = out["results"][p][label]
        cells.append(f"{r['mean_agreement']:.3f} / {r['match_frac']:.2f}")
    print(f"{label:<13}{ng:>7}" + "".join(f"{c:>22}" for c in cells), flush=True)
print("ALL DONE", flush=True)

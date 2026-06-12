"""Agreement of the Kilosort4 sort vs the MountainSort5 sort of the SAME 48 h
lossless (blosc) recording, on identical preprocessing (bandpass 300-6000 Hz +
global CMR, float32, the same materialized cache).

- reference: ``blosc-scheme2-train3600s`` -- MS5 scheme 2, single 48 h block, 1 h
  training (``27_``). Chosen as the reference because, like KS4, it sorts the whole
  48 h in one continuous pass with no per-block retraining (the structural analog of
  KS4); its own ``analyzer.zarr`` (``28_``) supplies the curation metrics.
- other:     ``blosc-ks4-nodrift`` -- Kilosort4, by tetrode group, no drift
  correction (``30_``).

So disagreement here is the sorter effect (MS5 vs KS4), holding the recording,
preprocessing, and continuous-pass structure fixed.

Reports raw (all reference units) and curated agreement. "Well-isolated" reference
units are defined on the reference (MS5) sort's OWN analyzer quality_metrics using
the metrics that discriminate on this 48 h data (ISI violations + RP contamination +
firing-rate floor; see 19_/20_) -- the SPOT choice.
"""
import json
import pathlib
import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
HERE = pathlib.Path(__file__).resolve().parent

REF_AGG = SR / "blosc-scheme2-train3600s" / "aggregated"          # MS5 scheme 2, single block
REF_ANALYZER = SR / "blosc-scheme2-train3600s" / "analyzer.zarr"  # curation metrics
NEW_AGG = SR / "blosc-ks4-nodrift" / "aggregated"                 # Kilosort4, no drift
REF_NAME = "ms5-singleblock-scheme2"
NEW_NAME = "ks4-nodrift"

TIER_THRESH = {  # (isi_violations_ratio <, rp_contamination <, firing_rate >=)
    "permissive": (0.5, 0.5, 0.2), "moderate": (0.3, 0.3, 0.5), "conservative": (0.1, 0.1, 0.5),
}

# ---- load both sortings + reference quality metrics ----
ref = si.load(str(REF_AGG))
new = si.load(str(NEW_AGG))
ref_ids = np.asarray(ref.unit_ids)
print(f"reference ({REF_NAME}): {ref.get_num_units()} units | "
      f"other ({NEW_NAME}): {new.get_num_units()} units", flush=True)

qm = si.load_sorting_analyzer(str(REF_ANALYZER)).get_extension("quality_metrics").get_data()
need_cols = {"isi_violations_ratio", "rp_contamination", "firing_rate"}
missing = need_cols - set(qm.columns)
if missing:
    raise SystemExit(f"reference quality_metrics missing columns {missing}; has {list(qm.columns)}")


def good_ids(tier):
    isi_t, rp_t, fr_t = TIER_THRESH[tier]
    ok = (qm["isi_violations_ratio"] < isi_t) & (qm["rp_contamination"] < rp_t) & (qm["firing_rate"] >= fr_t)
    ok &= np.isfinite(qm["isi_violations_ratio"]) & np.isfinite(qm["rp_contamination"]) & np.isfinite(qm["firing_rate"])
    ids = qm.index[ok].to_numpy()
    return np.array([i for i in ids if i in set(ref_ids.tolist())])


GOOD = {t: good_ids(t) for t in TIER_THRESH}
print("good-unit counts:", {t: int(len(v)) for t, v in GOOD.items()}, flush=True)

# ---- compare new (KS4) vs reference (MS5): best match per reference unit ----
cmp = sc.compare_two_sorters(ref, new, sorting1_name=REF_NAME, sorting2_name=NEW_NAME, match_score=0.5)
m = cmp.get_matching()[0]          # best match per reference unit (-1 if none)
ag = cmp.agreement_scores

results = {}
for label, ids in [("all_units", ref_ids), *[(t, GOOD[t]) for t in TIER_THRESH]]:
    ids = np.asarray([i for i in ids if i in set(ref_ids.tolist())])
    matched = [u for u in ids if m.get(u, -1) != -1]
    frac = len(matched) / len(ids) if len(ids) else float("nan")
    mean_ag = float(np.nanmean([ag.loc[u, m[u]] for u in matched])) if matched else float("nan")
    results[label] = {"n_ref": int(len(ids)), "matched": len(matched),
                      "match_frac": round(frac, 3), "mean_agreement": round(mean_ag, 4)}

out = {
    "reference": f"{REF_NAME} (blosc-scheme2-train3600s, MS5 scheme 2 / single block / 1 h train)",
    "other": f"{NEW_NAME} (blosc-ks4-nodrift, Kilosort4 by group / no drift)",
    "ref_total_units": int(ref.get_num_units()),
    "other_total_units": int(new.get_num_units()),
    "curation": "ISI<t & RP<t & FR>=t on reference's own (MS5) analyzer quality_metrics",
    "tier_thresholds": {t: {"isi<": v[0], "rp<": v[1], "fr>=": v[2]} for t, v in TIER_THRESH.items()},
    "n_good_units": {t: int(len(v)) for t, v in GOOD.items()},
    "good_unit_ids": {t: [int(x) for x in v] for t, v in GOOD.items()},
    "results": results,
}
(SR / "comparison_ks4_vs_ms5_summary.json").write_text(json.dumps(out, indent=2))

# ---- ready-to-read table ----
print("\n--- KILOSORT4 vs MOUNTAINSORT5 (single-block scheme 2): agreement vs MS5 reference ---", flush=True)
print(f"{'set':<13}{'n_ref':>7}{'matched':>9}{'match_frac':>12}{'mean_agree':>12}", flush=True)
for label in ["all_units", *TIER_THRESH]:
    r = results[label]
    print(f"{label:<13}{r['n_ref']:>7}{r['matched']:>9}{r['match_frac']:>12.3f}"
          f"{r['mean_agreement']:>12}", flush=True)
print(f"\nunit counts: {REF_NAME}={ref.get_num_units()} | {NEW_NAME}={new.get_num_units()}", flush=True)

# ---- ordered agreement-matrix figure (best matches on the diagonal) ----
scores = cmp.get_ordered_agreement_scores()
n_matched = int(results["all_units"]["matched"])


def _sparse_ticks(n, k=6):
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(scores.values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("agreement score (Jaccard)")
ax.set_xticks(_sparse_ticks(scores.shape[1]))
ax.set_xticklabels(_sparse_ticks(scores.shape[1]))
ax.set_yticks(_sparse_ticks(scores.shape[0]))
ax.set_yticklabels(_sparse_ticks(scores.shape[0]))
ax.set_xlabel(f"{NEW_NAME} unit (n={scores.shape[1]})")
ax.set_ylabel(f"{REF_NAME} unit (n={scores.shape[0]})")
ax.set_title("Agreement: Kilosort4 (no drift) vs MountainSort5 (single-block scheme 2)\n"
             f"{n_matched} matched units, "
             f"mean matched agreement = {results['all_units']['mean_agreement']}")
fig.tight_layout()
fig.savefig(HERE / "agreement_ks4_vs_ms5.png", dpi=150)
print("saved", HERE / "agreement_ks4_vs_ms5.png", flush=True)
print("ALL DONE", flush=True)

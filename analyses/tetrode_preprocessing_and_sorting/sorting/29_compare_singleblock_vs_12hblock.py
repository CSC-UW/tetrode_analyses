"""Agreement of the single-block scheme-2 sort vs the 12 h-block scheme-3 sort.

Both sortings are of the SAME 48 h lossless (blosc) recording, fixed pipeline (global
CMR, whitening_seed=42, deterministic PCA), differing only in the ms5 scheme/block
structure:
- reference: ``blosc-43200s-train3600s`` -- scheme 3, 12 h blocks, 1 h training
  (``24_``), 232 units;
- other:     ``blosc-scheme2-train3600s`` -- scheme 2, single 48 h block, 1 h training
  (``27_``).

So any disagreement here is the single-block-vs-blocked effect (the determinism
ceiling is 1.0 -- blosc-A vs blosc-B -- so it is not run-to-run noise).

Reports raw (all reference units) and curated agreement. "Well-isolated" reference
units are defined on the reference sort's OWN analyzer quality_metrics
(``blosc-43200s-train3600s/analyzer.zarr``, built by ``25_``) using the metrics that
discriminate on this 48 h data (ISI violations + RP contamination + firing-rate floor;
see 19_/20_) -- the SPOT choice, rather than the metric CSV of a different sort.
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

REF_AGG = SR / "blosc-43200s-train3600s" / "aggregated"          # scheme 3, 12 h blocks
REF_ANALYZER = SR / "blosc-43200s-train3600s" / "analyzer.zarr"  # curation metrics
NEW_AGG = SR / "blosc-scheme2-train3600s" / "aggregated"         # scheme 2, single block
REF_NAME = "12hblock"
NEW_NAME = "singleblock-scheme2"

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

# ---- compare new vs reference (best match per reference unit) ----
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
    "reference": f"{REF_NAME} (blosc-43200s-train3600s, scheme 3 / 12 h blocks / 1 h train)",
    "other": f"{NEW_NAME} (blosc-scheme2-train3600s, scheme 2 / single block / 1 h train)",
    "ref_total_units": int(ref.get_num_units()),
    "other_total_units": int(new.get_num_units()),
    "curation": "ISI<t & RP<t & FR>=t on reference's own analyzer quality_metrics",
    "tier_thresholds": {t: {"isi<": v[0], "rp<": v[1], "fr>=": v[2]} for t, v in TIER_THRESH.items()},
    "n_good_units": {t: int(len(v)) for t, v in GOOD.items()},
    "good_unit_ids": {t: [int(x) for x in v] for t, v in GOOD.items()},
    "results": results,
}
(SR / "comparison_singleblock_vs_12hblock_summary.json").write_text(json.dumps(out, indent=2))

# ---- ready-to-read table ----
print("\n--- SINGLE-BLOCK (scheme 2) vs 12 h-BLOCK (scheme 3): agreement vs reference ---", flush=True)
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
ax.set_title("Agreement: single-block scheme 2 vs 12 h-block scheme 3\n"
             f"{n_matched} matched units, "
             f"mean matched agreement = {results['all_units']['mean_agreement']}")
fig.tight_layout()
fig.savefig(HERE / "agreement_singleblock_vs_12hblock.png", dpi=150)
print("saved", HERE / "agreement_singleblock_vs_12hblock.png", flush=True)
print("ALL DONE", flush=True)

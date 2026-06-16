"""Refine the geometry-free UnitRefine retrain by dropping duration/length-confounded features.

Follows script 47. User observations (2026-06-12), generalized:
  * num_spikes is a RAW COUNT = firing_rate x duration -> perfectly collinear with firing_rate
    within our single 48h recording (zero added info) AND out-of-range vs short NP1.0 training
    sessions. Same hazard for the other raw counts isi_violations_count and rp_violations
    (each has a normalized counterpart already in the set). Prefer rate/ratio over count.
  * presence_ratio penalizes good units on a 48h recording where units drift in/out (our tracked
    units span part of 48h by design), while NP1.0 training units cluster near 1.0 -> adverse bias.

So drop {num_spikes, isi_violations_count, rp_violations, presence_ratio}: 28 -> 24 features.
Reports: (a) 5-fold CV on UnitRefine's own data, 28 vs 24 (in-distribution cost of dropping);
(b) evidence for the two intuitions on OUR data (num_spikes~firing_rate collinearity; conservative
units' presence_ratio distribution); (c) the 24-feature model applied to our units vs the 28-feat.

    cd gfys_workspace
    uv run --with skops --with huggingface_hub --extra tetrodes python \
      ../tetrode_analyses/.../sorting/48_unitrefine_drop_duration_confounded.py
"""
import pathlib

import numpy as np
import pandas as pd
import spikeinterface as si
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spikeinterface.curation.model_based_curation import ModelBasedClassification

REPO = "SpikeInterface/UnitRefine_noise_neural_classifier"
ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
ANALYZER = ROOT / "sortings_seed42_pcafix" / "tracked_48h" / "analyzer_clustered.zarr"
OUT_CSV = ANALYZER.parent / "unitrefine_24feat_noise_neural.csv"

DROP_GEOMETRY = ["drift_ptp", "drift_std", "drift_mad",
                 "spread", "velocity_above", "velocity_below", "exp_decay"]
DROP_NAN_HERE = ["amplitude_cv_median", "amplitude_cv_range"]
DROP_DURATION = ["num_spikes", "isi_violations_count", "rp_violations", "presence_ratio"]
LABELS = {0: "neural", 1: "noise"}
SEED = 42
RF_KW = dict(n_estimators=150, criterion="gini", min_samples_leaf=4, min_samples_split=3,
             class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def make_pipeline():
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("scale", StandardScaler()),
                     ("rf", RandomForestClassifier(**RF_KW))])


def cv(name, X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s = cross_validate(make_pipeline(), X, y, cv=skf, scoring=["accuracy", "balanced_accuracy"])
    print(f"  {name:24s} ({X.shape[1]:2d} feats)  acc={s['test_accuracy'].mean():.3f}  "
          f"bal_acc={s['test_balanced_accuracy'].mean():.3f}")


def apply_to_ours(feats_renamed, cols, pipe, tier, uids):
    X = feats_renamed[cols]
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X).max(axis=1)
    labels = np.array([LABELS[i] for i in pred])
    order = ["conservative", "moderate", "permissive", "none"]
    ct = pd.crosstab(pd.Categorical(tier, categories=order, ordered=True),
                     pd.Categorical(labels, categories=["neural", "noise"]),
                     rownames=["tier"], colnames=["label"], dropna=False)
    ct["n"] = ct.sum(axis=1)
    ct["neural_frac"] = (ct.get("neural", 0) / ct["n"]).round(3)
    return labels, proba, ct


def main():
    td = pd.read_csv(hf_hub_download(REPO, "training_data.csv"))
    y = pd.read_csv(hf_hub_download(REPO, "labels.csv")).iloc[:, -1].to_numpy()
    full = [c for c in td.columns if c != "unit_id"]
    feats28 = [c for c in full if c not in set(DROP_GEOMETRY + DROP_NAN_HERE)]
    feats24 = [c for c in feats28 if c not in set(DROP_DURATION)]

    print("== 5-fold CV on UnitRefine's own (NP1.0) data ==")
    cv("REDUCED geom-free", td[feats28], y)
    cv("minus duration-confound", td[feats24], y)
    print(f"  (dropped from 28->24: {DROP_DURATION})")

    # ---- evidence for the two intuitions, on OUR data ----
    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    uids = np.asarray(analyzer.sorting.unit_ids)
    tier = analyzer.sorting.get_property("tier")
    feats = analyzer.get_metrics_extension_data()
    mbc = ModelBasedClassification(analyzer, make_pipeline().fit(td[feats24], y))  # rename helper
    feats = mbc.handle_backwards_compatibility_in_metrics(
        feats, model_info={"requirements": {"spikeinterface": "0.102.0"}})

    ns, fr = feats["num_spikes"].to_numpy(float), feats["firing_rate"].to_numpy(float)
    ok = np.isfinite(ns) & np.isfinite(fr)
    print("\n== evidence on our 2204 units ==")
    print(f"num_spikes vs firing_rate Pearson r = {np.corrcoef(ns[ok], fr[ok])[0,1]:.4f} "
          f"(collinear within one fixed-duration recording -> num_spikes redundant)")
    pr = feats["presence_ratio"].to_numpy(float)
    for t in ["conservative", "moderate", "permissive"]:
        v = pr[tier == t]
        v = v[np.isfinite(v)]
        print(f"presence_ratio | {t:12s}: median={np.median(v):.2f}  "
              f"[p10={np.percentile(v,10):.2f}, p90={np.percentile(v,90):.2f}]  frac>=0.99={np.mean(v>=0.99):.2f}")

    # ---- apply both models to our units ----
    pipe28 = make_pipeline().fit(td[feats28], y)
    pipe24 = make_pipeline().fit(td[feats24], y)
    lab28, p28, ct28 = apply_to_ours(feats, feats28, pipe28, tier, uids)
    lab24, p24, ct24 = apply_to_ours(feats, feats24, pipe24, tier, uids)

    print("\n== 28-feature model (script 47) ==")
    print(ct28.to_string())
    print(f"confidence: median={np.median(p28):.3f} max={p28.max():.3f} >=0.7={ (p28>=0.7).mean():.1%}")
    print("\n== 24-feature model (drop duration-confounded) ==")
    print(ct24.to_string())
    print(f"confidence: median={np.median(p24):.3f} max={p24.max():.3f} >=0.7={ (p24>=0.7).mean():.1%}")
    iso = np.isin(tier, ["conservative", "moderate", "permissive"])
    print(f"\nwell-isolated called NOISE:  28-feat={ (lab28[iso]=='noise').mean():.1%}   "
          f"24-feat={ (lab24[iso]=='noise').mean():.1%}")
    flipped = int((lab28 != lab24).sum())
    print(f"units whose label changed 28->24: {flipped} "
          f"(neural->noise {int(((lab28=='neural')&(lab24=='noise')).sum())}, "
          f"noise->neural {int(((lab28=='noise')&(lab24=='neural')).sum())})")

    pd.DataFrame({"unit_id": uids, "tier": tier,
                  "label_24feat": lab24, "probability_24feat": p24}).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\nDONE")


if __name__ == "__main__":
    main()

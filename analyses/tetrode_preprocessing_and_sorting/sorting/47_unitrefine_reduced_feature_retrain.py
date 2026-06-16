"""Retrain UnitRefine's noise/neural classifier on geometry-free features only, then apply.

Motivation: the stock UnitRefine model (script 46) is near-chance on our tetrodes because
~11/37 features are geometry-dependent/degenerate (drift x3, spread, velocity x2, exp_decay,
amplitude_cv x2, ...). You cannot drop features from an ALREADY-FITTED forest, but UnitRefine
publishes its training data, so we can RETRAIN on the applicable subset and measure the cost.

Experiment:
  1. Download UnitRefine's published training_data.csv + labels.csv (NP1.0 mouse V1/SC/ALM;
     5121 units, 3690 neural / 1431 noise).
  2. 5-fold stratified CV: FULL (37 feats) vs REDUCED (28 geometry-free feats), replicating
     the published best RF architecture. Measures the discriminative cost of dropping.
  3. Report impurity feature importances on the reduced model (which features carry signal).
  4. Fit the reduced model on all training data, apply to our 2204 tetrode units, and compare
     confidence + isolation-tier agreement against the stock model (script 46).

DROPPED (not applicable on fictional tetrode geometry / not computable here):
  geometry: drift_ptp, drift_std, drift_mad, spread, velocity_above, velocity_below, exp_decay
  all-NaN on our analyzer (recomputable, geometry-free): amplitude_cv_median, amplitude_cv_range

    cd gfys_workspace
    uv run --with skops --with huggingface_hub --extra tetrodes python \
      ../tetrode_analyses/.../sorting/47_unitrefine_reduced_feature_retrain.py
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
OUT_CSV = ANALYZER.parent / "unitrefine_reduced_noise_neural.csv"

DROP_GEOMETRY = ["drift_ptp", "drift_std", "drift_mad",
                 "spread", "velocity_above", "velocity_below", "exp_decay"]
DROP_NAN_HERE = ["amplitude_cv_median", "amplitude_cv_range"]
LABELS = {0: "neural", 1: "noise"}
SEED = 42

# Published best RF (model_accuracies.csv row model_id=3): most_frequent + StandardScaler.
RF_KW = dict(n_estimators=150, criterion="gini", min_samples_leaf=4, min_samples_split=3,
             class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(**RF_KW)),
    ])


def cv_report(name, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    s = cross_validate(make_pipeline(), X, y, cv=cv,
                       scoring=["accuracy", "balanced_accuracy", "precision", "recall"])
    print(f"  {name:20s} ({X.shape[1]:2d} feats)  "
          f"acc={s['test_accuracy'].mean():.3f}  bal_acc={s['test_balanced_accuracy'].mean():.3f}  "
          f"prec={s['test_precision'].mean():.3f}  rec={s['test_recall'].mean():.3f}")
    return s


def main():
    td = pd.read_csv(hf_hub_download(REPO, "training_data.csv"))
    lb = pd.read_csv(hf_hub_download(REPO, "labels.csv"))
    y = lb[lb.columns[-1]].to_numpy()
    full_feats = [c for c in td.columns if c != "unit_id"]
    reduced_feats = [c for c in full_feats if c not in set(DROP_GEOMETRY + DROP_NAN_HERE)]

    print(f"training: {td.shape[0]} units, {len(full_feats)} full feats -> "
          f"{len(reduced_feats)} reduced (dropped {len(full_feats)-len(reduced_feats)}); "
          f"labels {int((y==0).sum())} neural / {int((y==1).sum())} noise\n")

    print("== 5-fold CV on UnitRefine's own data (replicated best RF) ==")
    cv_report("FULL", td[full_feats], y)
    cv_report("REDUCED geom-free", td[reduced_feats], y)

    # Fit reduced on all training data; impurity importances
    pipe = make_pipeline().fit(td[reduced_feats], y)
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=reduced_feats).sort_values(ascending=False)
    print("\n== reduced-model feature importances (top 12) ==")
    for k, v in imp.head(12).items():
        print(f"  {k:24s} {v:.3f}")

    # Apply to our tetrode units (reduced features only)
    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    uids = np.asarray(analyzer.sorting.unit_ids)
    tier = analyzer.sorting.get_property("tier")
    feats = analyzer.get_metrics_extension_data()
    mbc = ModelBasedClassification(analyzer, pipe)  # for the rename helper only
    feats = mbc.handle_backwards_compatibility_in_metrics(
        feats, model_info={"requirements": {"spikeinterface": "0.102.0"}})
    missing = [c for c in reduced_feats if c not in feats.columns]
    assert not missing, f"reduced features missing on our data: {missing}"
    X_ours = feats[reduced_feats]
    pred = pipe.predict(X_ours)
    proba = pipe.predict_proba(X_ours).max(axis=1)
    labels = np.array([LABELS[i] for i in pred])

    df = pd.DataFrame({"unit_id": uids, "tier": tier,
                       "classifier_label": labels, "probability": proba})
    df.to_csv(OUT_CSV, index=False)

    n = len(df)
    print(f"\n== reduced model applied to {n} tetrode units ==")
    print(f"neural={int((labels=='neural').sum())} ({(labels=='neural').mean():.1%})  "
          f"noise={int((labels=='noise').sum())} ({(labels=='noise').mean():.1%})")
    p = proba
    print(f"confidence: median={np.median(p):.3f} max={p.max():.3f} | "
          f">=0.7: {(p>=0.7).mean():.1%}  >=0.9: {(p>=0.9).mean():.1%}  "
          f"(stock model: median 0.58, max 0.88, none>=0.9)")

    print("\nclassifier label x isolation tier:")
    order = ["conservative", "moderate", "permissive", "none"]
    ct = pd.crosstab(pd.Categorical(tier, categories=order, ordered=True),
                     pd.Categorical(labels, categories=["neural", "noise"]),
                     rownames=["tier"], colnames=["label"], dropna=False)
    ct["n"] = ct.sum(axis=1)
    ct["neural_frac"] = (ct.get("neural", 0) / ct["n"]).round(3)
    print(ct.to_string())
    iso = np.isin(tier, ["conservative", "moderate", "permissive"])
    print(f"\nwell-isolated (>=permissive) called NOISE: "
          f"{int((labels[iso]=='noise').sum())}/{int(iso.sum())} "
          f"({(labels[iso]=='noise').mean():.1%})   (stock model: 59.9%)")
    print(f"\nwrote {OUT_CSV}\nDONE")


if __name__ == "__main__":
    main()

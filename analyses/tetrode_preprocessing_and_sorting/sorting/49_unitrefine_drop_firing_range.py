"""Follow-up to script 48: also drop firing_range (24 -> 23 features).

firing_range = spread (5-95 pctile) of per-bin firing rate across the recording. Over our 48h
sleep-deprivation recording spanning NREM/REM/wake (+ SD), neurons legitimately show large
firing-rate excursions that short, more-stationary NP1.0 training sessions would not -> a
recording-length/state confound (different mechanism than the raw counts, same "long & non-
stationary" concern). This script measures the in-distribution cost and the effect on our units,
and shows the ours-vs-training firing_range distribution shift as evidence.

    cd gfys_workspace
    uv run --with skops --with huggingface_hub --extra tetrodes python \
      ../tetrode_analyses/.../sorting/49_unitrefine_drop_firing_range.py
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
OUT_CSV = ANALYZER.parent / "unitrefine_23feat_noise_neural.csv"

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
    print(f"  {name:26s} ({X.shape[1]:2d} feats)  acc={s['test_accuracy'].mean():.3f}  "
          f"bal_acc={s['test_balanced_accuracy'].mean():.3f}")


def apply_to_ours(feats_renamed, cols, pipe, tier):
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
    feats24 = [c for c in full if c not in set(DROP_GEOMETRY + DROP_NAN_HERE + DROP_DURATION)]
    feats23 = [c for c in feats24 if c != "firing_range"]

    print("== 5-fold CV on UnitRefine's own (NP1.0) data ==")
    cv("minus duration-confound", td[feats24], y)
    cv("also minus firing_range", td[feats23], y)

    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    tier = analyzer.sorting.get_property("tier")
    uids = np.asarray(analyzer.sorting.unit_ids)
    feats = analyzer.get_metrics_extension_data()
    mbc = ModelBasedClassification(analyzer, make_pipeline().fit(td[feats23], y))
    feats = mbc.handle_backwards_compatibility_in_metrics(
        feats, model_info={"requirements": {"spikeinterface": "0.102.0"}})

    # evidence: firing_range distribution, training (NP1.0) vs ours
    fr_tr = td["firing_range"].to_numpy(float)
    fr_tr = fr_tr[np.isfinite(fr_tr)]
    fr_us = feats["firing_range"].to_numpy(float)
    fr_us = fr_us[np.isfinite(fr_us)]
    print("\n== firing_range distribution shift (Hz) ==")
    print(f"  NP1.0 training: median={np.median(fr_tr):.2f}  p90={np.percentile(fr_tr,90):.2f}  max={fr_tr.max():.1f}")
    print(f"  our tetrodes  : median={np.median(fr_us):.2f}  p90={np.percentile(fr_us,90):.2f}  max={fr_us.max():.1f}")
    for t in ["conservative", "moderate", "permissive"]:
        v = feats["firing_range"].to_numpy(float)[tier == t]
        v = v[np.isfinite(v)]
        print(f"  our {t:12s}: median={np.median(v):.2f}  p90={np.percentile(v,90):.2f}")

    pipe24 = make_pipeline().fit(td[feats24], y)
    pipe23 = make_pipeline().fit(td[feats23], y)
    lab24, p24, ct24 = apply_to_ours(feats, feats24, pipe24, tier)
    lab23, p23, ct23 = apply_to_ours(feats, feats23, pipe23, tier)

    print("\n== 24-feature model (script 48) ==")
    print(ct24.to_string())
    print(f"confidence: median={np.median(p24):.3f} max={p24.max():.3f} >=0.7={(p24>=0.7).mean():.1%}")
    print("\n== 23-feature model (also drop firing_range) ==")
    print(ct23.to_string())
    print(f"confidence: median={np.median(p23):.3f} max={p23.max():.3f} >=0.7={(p23>=0.7).mean():.1%}")
    iso = np.isin(tier, ["conservative", "moderate", "permissive"])
    print(f"\nwell-isolated called NOISE:  24-feat={(lab24[iso]=='noise').mean():.1%}   "
          f"23-feat={(lab23[iso]=='noise').mean():.1%}")
    print(f"overall neural frac:  24-feat={(lab24=='neural').mean():.1%}   23-feat={(lab23=='neural').mean():.1%}")
    print(f"units whose label changed 24->23: {int((lab24!=lab23).sum())} "
          f"(neural->noise {int(((lab24=='neural')&(lab23=='noise')).sum())}, "
          f"noise->neural {int(((lab24=='noise')&(lab23=='neural')).sum())})")

    pd.DataFrame({"unit_id": uids, "tier": tier,
                  "label_23feat": lab23, "probability_23feat": p23}).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\nDONE")


if __name__ == "__main__":
    main()

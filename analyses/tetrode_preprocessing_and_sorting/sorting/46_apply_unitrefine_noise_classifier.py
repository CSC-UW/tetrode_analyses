"""Apply UnitRefine's pretrained noise/neural classifier to the tracked analyzer.

Model: ``SpikeInterface/UnitRefine_noise_neural_classifier`` (HuggingFace). A
scikit-learn pipeline (SimpleImputer -> StandardScaler -> RandomForest) trained on
**Neuropixels 1.0, mouse V1/SC/ALM** quality+template metrics. label_conversion:
0=neural, 1=noise.

HEAVY CAVEAT (out-of-distribution): our data are 16 tetrodes with FICTIONAL geometry
and 4-channel sparse waveforms. Consequences baked into this run:
  * 3/37 model features are the spike-location DRIFT metrics (drift_ptp/std/mad). They
    require geometry, so our geometry-free analyzer has none -> injected as NaN and
    filled by the pipeline's median imputer (i.e. an NP1.0 prior constant).
  * geometry-dependent TEMPLATE metrics (spread, velocity_above/below, exp_decay) are
    computed on fictional geometry -> not physically meaningful for tetrodes, yet feed
    the model.
  * sklearn 1.4.2 (train) vs 1.8.0 (here) -> InconsistentVersionWarning (benign for RF).
So treat the noise/neural call as ADVISORY, cross-checked against our isolation tiers,
NOT as ground truth.

This script is NON-DESTRUCTIVE: it drives the pipeline through the library's own
helper methods but does NOT call predict_labels (which would persist
classifier_label/classifier_probability into the zarr via set_sorting_property(save=True
default)). Pass --persist to write those two columns into analyzer_clustered.zarr.

    cd gfys_workspace
    uv run --with skops --with huggingface_hub --extra tetrodes python \
      ../tetrode_analyses/.../sorting/46_apply_unitrefine_noise_classifier.py [--persist]
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import spikeinterface as si
from spikeinterface.curation import load_model
from spikeinterface.curation.model_based_curation import ModelBasedClassification
from spikeinterface.curation.train_manual_curation import _format_metric_dataframe

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
ANALYZER = ROOT / "sortings_seed42_pcafix" / "tracked_48h" / "analyzer_clustered.zarr"
OUT_CSV = ANALYZER.parent / "unitrefine_noise_neural.csv"
REPO = "SpikeInterface/UnitRefine_noise_neural_classifier"
DRIFT_METRICS = ["drift_ptp", "drift_std", "drift_mad"]  # geometry-dependent -> unavailable


def _patch_sklearn_compat(pipeline):
    """Backfill attributes lost unpickling 1.4.2 estimators under sklearn 1.8.

    SimpleImputer.transform in 1.8 reads ``self._fill_dtype`` (set at fit-time to the
    input dtype). A 1.4.2 pickle lacks it; our metric matrix is float64, so restore it.
    """
    for _, step in getattr(pipeline, "steps", []):
        if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = np.dtype(np.float64)


def classify(analyzer, persist=False):
    pipeline, model_info = load_model(repo_id=REPO, trusted=["numpy.dtype"])
    _patch_sklearn_compat(pipeline)
    mbc = ModelBasedClassification(analyzer, pipeline)

    # Build the metrics frame the library would use, then inject the unavailable drift
    # columns as NaN so the required-metric check passes (imputer fills them downstream).
    feats = analyzer.get_metrics_extension_data()
    for m in DRIFT_METRICS:
        if m not in feats.columns:
            feats[m] = np.nan

    # Library handles the <0.103.2 template-metric renames (peak_to_trough_duration->
    # peak_to_valley, peak_after_to_trough_ratio->peak_trough_ratio[sign-flip],
    # trough_half_width->half_width), then selects+orders the 37 required features.
    feats = mbc.handle_backwards_compatibility_in_metrics(feats, model_info=model_info)
    feats = mbc._check_required_metrics_are_present(feats)
    X = _format_metric_dataframe(feats)

    pred_int = pipeline.predict(X)
    proba = pipeline.predict_proba(X).max(axis=1)
    conv = {int(k): v for k, v in model_info["label_conversion"].items()}
    labels = np.array([conv[i] for i in pred_int])

    if persist:
        analyzer.set_sorting_property("classifier_label", labels, save=True)
        analyzer.set_sorting_property("classifier_probability", proba.astype(np.float32), save=True)

    return labels, proba, len(mbc.required_metrics)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist", action="store_true",
                    help="write classifier_label/classifier_probability into the analyzer zarr")
    args = ap.parse_args()

    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    uids = np.asarray(analyzer.sorting.unit_ids)
    tier = analyzer.sorting.get_property("tier")
    n_chunks = analyzer.sorting.get_property("n_chunks")

    labels, proba, n_feat = classify(analyzer, persist=args.persist)

    df = pd.DataFrame({
        "unit_id": uids, "tier": tier, "n_chunks": n_chunks,
        "classifier_label": labels, "probability": proba,
    })
    df.to_csv(OUT_CSV, index=False)

    n = len(df)
    n_neural = int((labels == "neural").sum())
    n_noise = int((labels == "noise").sum())
    print(f"model: {REPO}  ({n_feat} features; drift x3 imputed; geometry-free tetrode data)")
    print(f"n_units={n}  neural={n_neural} ({n_neural/n:.1%})  noise={n_noise} ({n_noise/n:.1%})")
    print(f"mean prob: neural={proba[labels=='neural'].mean():.3f}  noise={proba[labels=='noise'].mean():.3f}")

    print("\n== classifier label x isolation tier ==")
    tier_order = ["conservative", "moderate", "permissive", "none"]
    ct = pd.crosstab(pd.Categorical(tier, categories=tier_order, ordered=True),
                     pd.Categorical(labels, categories=["neural", "noise"]),
                     rownames=["tier"], colnames=["label"], dropna=False)
    ct["n"] = ct.sum(axis=1)
    ct["neural_frac"] = (ct.get("neural", 0) / ct["n"]).round(3)
    print(ct.to_string())

    # Among our well-isolated (>=permissive) units, how many does the model call noise?
    iso = np.isin(tier, ["conservative", "moderate", "permissive"])
    if iso.any():
        noise_in_iso = (labels[iso] == "noise").sum()
        print(f"\nwell-isolated (>=permissive) units called NOISE by model: "
              f"{noise_in_iso}/{iso.sum()} ({noise_in_iso/iso.sum():.1%})")
    print(f"\nwrote {OUT_CSV}")
    if args.persist:
        print("PERSISTED classifier_label/classifier_probability into analyzer.zarr")
    print("DONE")


if __name__ == "__main__":
    main()

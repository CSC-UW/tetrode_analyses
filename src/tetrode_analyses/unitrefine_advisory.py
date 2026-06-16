"""Advisory noise/neural ranking for tetrode sortings, derived from UnitRefine.

The stock ``SpikeInterface/UnitRefine_noise_neural_classifier`` (trained on Neuropixels 1.0
mouse cortex) is out-of-distribution for our fictional-geometry tetrodes: ~11/37 of its features
are geometry-dependent/degenerate, and it is near-chance on our data (see the sorting analyses
scripts 46-49 and docs). This module instead **retrains** the classifier on UnitRefine's own
published training data using only the features that are (a) geometry-free and (b) not confounded
by recording length -- a basis that transfers far better to tetrodes.

WHAT IT IS / IS NOT
-------------------
* It IS a useful **advisory ranking**: P(neural) orders units, and that order agrees strongly with
  our independent isolation-tier gate (well-isolated units score most-neural). Use it to triage /
  corroborate, e.g. sort the curation unit list by ``unitrefine_neural_prob``.
* It is NOT a calibrated decision rule on tetrode data. The probabilities remain near 0.5 (cross-
  domain shift: the same model is confident in-distribution, median ~0.93, but ~0.55 on tetrodes),
  so DO NOT hard-threshold at 0.5 to cut units. The label column is the 0.5 call kept only as a
  convenience; trust the ranking, not the cutoff. A calibrated model needs hand-labeled TETRODE
  training units (this same feature basis is the right input for that).

Feature lineage (see scripts 47/48/49 for the measurements behind each drop):
* drop 7 GEOMETRY features (drift x3, spread, velocity x2, exp_decay) -- not defined w/o geometry.
* drop 2 ALL-NaN-on-tetrode features (amplitude_cv_median/range) -- recomputable but absent here.
* drop 4 DURATION/COUNT-confounded (num_spikes, isi_violations_count, rp_violations -- raw counts
  redundant with firing_rate / isi_violations_ratio / rp_contamination; presence_ratio -- low by
  design for tracked units on a 48 h recording).
* drop firing_range -- state/length-confounded over a 48 h sleep-dep recording.
-> 23 features (``RECOMMENDED_FEATURES``); in-distribution CV ~0.91, ~free vs the full 37.

The trained model + a copy of the training CSVs are cached under ``cache_dir`` (default
``$TETRODE_MODELS_DIR`` or ``~/.cache/tetrode_analyses/unitrefine_advisory``). Robust by design:
loading a cached fitted model under a different sklearn version auto-retrains from the cached CSVs,
so it never breaks the way the original pickled skops model did. Only ``huggingface_hub`` (a one-
time download of the training CSVs) is an extra dependency; everything else is sklearn + joblib.
"""
from __future__ import annotations

import json
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

REPO = "SpikeInterface/UnitRefine_noise_neural_classifier"
SEED = 42
LABEL_CONVERSION = {0: "neural", 1: "noise"}

# Replicated published best RF (model_accuracies.csv, model_id=3): most_frequent impute +
# StandardScaler. Pinned so retraining is deterministic across machines.
RF_PARAMS = dict(n_estimators=150, criterion="gini", min_samples_leaf=4, min_samples_split=3,
                 class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)

# UnitRefine's full 37-feature input (pipeline.feature_names_in_), with our drop lists.
ALL_FEATURES = [
    "amplitude_cutoff", "amplitude_cv_median", "amplitude_cv_range", "amplitude_median",
    "drift_ptp", "drift_std", "drift_mad", "firing_range", "firing_rate",
    "isi_violations_ratio", "isi_violations_count", "num_spikes", "presence_ratio",
    "rp_contamination", "rp_violations", "sliding_rp_violation", "snr",
    "sync_spike_2", "sync_spike_4", "sync_spike_8", "d_prime", "isolation_distance",
    "l_ratio", "silhouette", "nn_hit_rate", "nn_miss_rate", "exp_decay", "half_width",
    "num_negative_peaks", "num_positive_peaks", "peak_to_valley", "peak_trough_ratio",
    "recovery_slope", "repolarization_slope", "spread", "velocity_above", "velocity_below",
]
DROP_GEOMETRY = ["drift_ptp", "drift_std", "drift_mad",
                 "spread", "velocity_above", "velocity_below", "exp_decay"]
DROP_NAN_ON_TETRODE = ["amplitude_cv_median", "amplitude_cv_range"]
DROP_DURATION_CONFOUND = ["num_spikes", "isi_violations_count", "rp_violations", "presence_ratio"]
DROP_FIRING_RANGE = ["firing_range"]

GEOMETRY_FREE_FEATURES = [c for c in ALL_FEATURES
                          if c not in set(DROP_GEOMETRY + DROP_NAN_ON_TETRODE)]  # 28
RECOMMENDED_FEATURES = [c for c in GEOMETRY_FREE_FEATURES
                        if c not in set(DROP_DURATION_CONFOUND + DROP_FIRING_RANGE)]  # 23

# Template-metric renames SpikeInterface applies for models declaring SI < 0.103.2 (our analyzers
# use the newer names). Mirrors curation.model_based_curation.handle_backwards_compatibility_in_metrics.
_TEMPLATE_RENAMES = {  # current_name -> training_name (+ optional sign flip)
    "peak_to_trough_duration": ("peak_to_valley", 1.0),
    "peak_after_to_trough_ratio": ("peak_trough_ratio", -1.0),
    "trough_half_width": ("half_width", 1.0),
}


def _default_cache_dir() -> pathlib.Path:
    env = os.environ.get("TETRODE_MODELS_DIR")
    base = pathlib.Path(env) if env else pathlib.Path.home() / ".cache" / "tetrode_analyses"
    return base / "unitrefine_advisory"


def _build_pipeline():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("scale", StandardScaler()),
                     ("rf", RandomForestClassifier(**RF_PARAMS))])


def _training_data(cache_dir: pathlib.Path):
    """Load the UnitRefine training CSVs from cache, downloading + caching them on first use."""
    td_path, lb_path = cache_dir / "training_data.csv", cache_dir / "labels.csv"
    if not (td_path.exists() and lb_path.exists()):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "Training CSVs are not cached and `huggingface_hub` is not installed to fetch them. "
                "Install the `advisory` extra (uv sync --extra advisory) or place training_data.csv "
                f"+ labels.csv in {cache_dir}."
            ) from e
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.read_csv(hf_hub_download(REPO, "training_data.csv")).to_csv(td_path, index=False)
        pd.read_csv(hf_hub_download(REPO, "labels.csv")).to_csv(lb_path, index=False)
    td = pd.read_csv(td_path)
    lb = pd.read_csv(lb_path)
    return td, lb.iloc[:, -1].to_numpy()


def train_model(cache_dir=None, features=RECOMMENDED_FEATURES, *, save=True):
    """Fit the advisory model on UnitRefine's data with `features`; optionally cache it. Returns (pipe, meta)."""
    import joblib
    cache_dir = pathlib.Path(cache_dir) if cache_dir else _default_cache_dir()
    td, y = _training_data(cache_dir)
    pipe = _build_pipeline().fit(td[list(features)], y)
    meta = {"repo": REPO, "n_features": len(features), "features": list(features),
            "label_conversion": LABEL_CONVERSION, "rf_params": {k: v for k, v in RF_PARAMS.items()},
            "seed": SEED, "n_train": int(len(y)),
            "advisory_only": True,
            "note": "Advisory ranking for tetrodes (out-of-distribution; probabilities uncalibrated). "
                    "Rank by P(neural); do NOT hard-threshold at 0.5."}
    if save:
        cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, cache_dir / "model.joblib")
        (cache_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return pipe, meta


def load_model(cache_dir=None, features=RECOMMENDED_FEATURES):
    """Return a fitted advisory pipeline: load the cached model, else train+cache it.

    If a cached model fails to load (e.g. a sklearn version bump, as happened to the original
    pickled skops model), it is transparently retrained from the cached CSVs.
    """
    import joblib
    cache_dir = pathlib.Path(cache_dir) if cache_dir else _default_cache_dir()
    model_path = cache_dir / "model.joblib"
    if model_path.exists():
        try:
            pipe = joblib.load(model_path)
            pipe.predict(pd.DataFrame([{f: 0.0 for f in features}]))  # smoke-test under current sklearn
            return pipe
        except Exception as e:  # version drift etc. -> retrain
            warnings.warn(f"cached advisory model failed to load ({e!r}); retraining from cached data")
    return train_model(cache_dir, features)[0]


def _prepare_features(analyzer, features):
    """Build the feature matrix (training-name columns) from an analyzer's quality+template metrics.

    The template renames mirror SpikeInterface's own
    ``curation.model_based_curation.handle_backwards_compatibility_in_metrics`` (replicated here to
    avoid coupling to a private method); they map our newer template-metric names back to the
    training names the model expects.
    """
    for ext in ("quality_metrics", "template_metrics"):
        if analyzer.get_extension(ext) is None:
            raise RuntimeError(f"analyzer is missing the '{ext}' extension required for advisory labeling")
    feats = analyzer.get_metrics_extension_data().copy()
    for cur, (new, sign) in _TEMPLATE_RENAMES.items():
        if cur in feats.columns:
            feats[new] = sign * feats[cur]
    missing = [f for f in features if f not in feats.columns]
    if missing:  # keep going; the imputer fills them (advisory tool, not a gate)
        warnings.warn(f"advisory features absent on this analyzer, imputed: {missing}")
        for f in missing:
            feats[f] = np.nan
    return feats[list(features)]


def label_units(analyzer, *, cache_dir=None, features=RECOMMENDED_FEATURES, persist=False,
                prefix="unitrefine"):
    """Apply the advisory model to a SortingAnalyzer; return a DataFrame; optionally persist columns.

    Returns columns: ``{prefix}_neural_prob`` (P(neural), the rankable advisory score in [0,1]) and
    ``{prefix}_label`` ("neural"/"noise", the 0.5 call -- advisory only). With ``persist=True`` these
    two are written onto the analyzer's sorting via ``set_sorting_property(save=True)`` so they travel
    with the analyzer (and the curation bundle) as sortable unit-list columns.
    """
    import spikeinterface as si
    if not hasattr(analyzer, "sorting"):
        analyzer = si.load_sorting_analyzer(str(analyzer))
    pipe = load_model(cache_dir, features)
    X = _prepare_features(analyzer, features)
    proba = pipe.predict_proba(X)
    neural_idx = list(pipe.classes_).index(0)  # class 0 == neural
    neural_prob = proba[:, neural_idx].astype(np.float32)
    labels = np.where(neural_prob >= 0.5, "neural", "noise")

    prob_col, label_col = f"{prefix}_neural_prob", f"{prefix}_label"
    if persist:
        analyzer.set_sorting_property(label_col, labels, save=True)
        analyzer.set_sorting_property(prob_col, neural_prob, save=True)
    return pd.DataFrame({"unit_id": np.asarray(analyzer.sorting.unit_ids),
                         prob_col: neural_prob, label_col: labels})

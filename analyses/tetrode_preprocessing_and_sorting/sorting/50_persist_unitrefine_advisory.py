"""Persist the final advisory UnitRefine noise/neural labels onto the tracked analyzer.

Uses the reusable `tetrode_analyses.unitrefine_advisory` model (23 geometry-free, duration-robust
features; see scripts 47-49 for the lineage). Writes two sortable unit properties onto
analyzer_clustered.zarr:
  * ``unitrefine_neural_prob`` (float32, P(neural) in [0,1]) -- the ADVISORY RANKING score.
  * ``unitrefine_label``       (str, "neural"/"noise")      -- the 0.5 call, advisory only.
ADVISORY ONLY: probabilities are uncalibrated on tetrode data (~0.55); rank by neural_prob, do NOT
hard-threshold to cut units.

The first run downloads + caches UnitRefine's training CSVs and fits/caches the model under
$TETRODE_MODELS_DIR (or ~/.cache/tetrode_analyses/unitrefine_advisory); later runs / other datasets
load the cached model. Idempotent (overwrites the two columns).

    cd gfys_workspace
    uv run --with huggingface_hub --extra tetrodes python \
      ../tetrode_analyses/.../sorting/50_persist_unitrefine_advisory.py
"""
import pathlib

import numpy as np
import pandas as pd
import spikeinterface as si

from tetrode_analyses import unitrefine_advisory as ura

ANALYZER = pathlib.Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/"
    "sortings_seed42_pcafix/tracked_48h/analyzer_clustered.zarr"
)


def main():
    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    ura.label_units(analyzer, persist=True)  # default 23-feature RECOMMENDED set; persists 2 columns

    # verify persisted + summarize against isolation tiers
    re = si.load_sorting_analyzer(str(ANALYZER))
    for col in ("unitrefine_label", "unitrefine_neural_prob"):
        assert re.sorting.get_property(col) is not None, f"{col} did not persist"
    lab = re.sorting.get_property("unitrefine_label")
    prob = np.asarray(re.sorting.get_property("unitrefine_neural_prob"), dtype=float)
    tier = re.sorting.get_property("tier")

    print(f"persisted unitrefine_label + unitrefine_neural_prob onto {ANALYZER.name}")
    print(f"model: {len(ura.RECOMMENDED_FEATURES)} features, cache -> {ura._default_cache_dir()}")
    print(f"advisory labels: neural={int((lab=='neural').sum())}  noise={int((lab=='noise').sum())}")
    print(f"P(neural): median={np.median(prob):.3f}  min={prob.min():.3f}  max={prob.max():.3f}")

    order = ["conservative", "moderate", "permissive", "none"]
    ct = pd.crosstab(pd.Categorical(tier, categories=order, ordered=True),
                     pd.Categorical(lab, categories=["neural", "noise"]),
                     rownames=["tier"], colnames=["label"], dropna=False)
    ct["median_P(neural)"] = [np.median(prob[tier == t]) if (tier == t).any() else np.nan for t in order]
    print("\nadvisory label x isolation tier (+ median P(neural), the ranking score):")
    print(ct.to_string())
    print("\nDONE")


if __name__ == "__main__":
    main()

"""Unit tests for the three-axis scoreboard aggregator (_scoreboard).

Verifies the thin aggregation layer -- spike-weighted axis-B purity, axis-C duplicate-pair counting, and
the head-to-head comparison table -- on synthetic inputs (no NFS). The underlying metrics are tested in
test_assignment_eval.py; here we only check that the scoreboard wires + summarises them correctly.
"""
import pathlib
import sys

import numpy as np

_SORTING_DIR = (pathlib.Path(__file__).resolve().parents[1]
                / "analyses" / "tetrode_preprocessing_and_sorting" / "sorting")
sys.path.insert(0, str(_SORTING_DIR))

import spikeinterface as si  # noqa: E402

from _scoreboard import axis_b_summary, axis_c_summary, compare_scoreboards  # noqa: E402

FS = 30000.0
WIN = 10000  # small CCG window so synthetic trains span >= 2 windows


def _spikes(clusters):
    arr = np.zeros(len(clusters), dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    arr["sample_index"] = np.arange(len(clusters)) * 50 + 200
    arr["cluster_index"] = clusters
    return arr


def _three_unit_sorting():
    # u0 base; u1 = u0 + 300 (refractory-dip duplicate of u0); u2 independent. All on tetrode 0.
    base = np.concatenate([np.arange(40) * 200 + w * WIN + 500 for w in range(3)]).astype(np.int64)
    rng = np.random.default_rng(0)
    indep = np.sort(np.concatenate(
        [rng.integers(w * WIN + 200, w * WIN + WIN - 200, size=120) for w in range(3)])).astype(np.int64)
    sort = si.NumpySorting.from_unit_dict([{0: base, 1: base + 300, 2: indep}], sampling_frequency=FS)
    sort.set_property("group", np.zeros(3, dtype=int))
    return sort


def test_axis_b_summary_spikeweighted_purity():
    sort = _three_unit_sorting()
    # u0: 10 spikes, 6 best-match self + 4 best-match u1; u1/u2: all self -> spike-weighted = 16/20 = 0.8
    clusters = np.array([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int64)
    rF_arg = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [1] * 5 + [2] * 5, dtype=np.int64)
    cosines = {"rF_arg": rF_arg, "rA_arg": rF_arg}
    out = axis_b_summary(_spikes(clusters), cosines, np.array([0, 1, 2]), sort, win_s=WIN / FS)

    assert out["n_units"] == 3
    assert out["spikeweighted_purity"] == np.float64(0.8)
    assert out["median_best_match_frac"] == 1.0  # median over [0.6, 1.0, 1.0]
    assert out["cosine_purity"][0]["top_neighbor"] == 1


def test_axis_c_summary_counts_duplicate_pair():
    sort = _three_unit_sorting()
    out = axis_c_summary(sort, win_s=WIN / FS)
    assert out["n_units"] == 3
    assert out["max_units_per_tetrode"] == 3
    assert out["n_ccg_duplicate_pairs"] >= 1  # the (u0, u1) +300-frame hand-off is a refractory dip
    assert out["ccg_skipped"] is False


def test_axis_c_skips_full_ccg_when_too_many_units():
    sort = _three_unit_sorting()
    out = axis_c_summary(sort, win_s=WIN / FS, max_units_for_full_ccg=2)
    assert out["ccg_skipped"] is True
    assert out["n_ccg_duplicate_pairs"] is None  # skip is reported, not silent


def test_compare_scoreboards_renders_table():
    results = {
        "MP": {"label": "MP", "n_units": 95,
               "axis_A": {">=10_pooled": 64.0, ">=12_pooled": 82.9},
               "axis_B": {"spikeweighted_purity": 0.91, "median_best_match_frac": 0.95,
                          "n_flagged": 4, "n_cross_contaminated": 2},
               "axis_C": {"n_ccg_duplicate_pairs": 3, "max_units_per_tetrode": 10}},
        "chunk+match": {"label": "chunk+match", "n_units": 2204,
                        "axis_A": {">=10_pooled": 99.5, ">=12_pooled": 99.5},
                        "axis_B": {"spikeweighted_purity": float("nan"), "median_best_match_frac": float("nan"),
                                   "n_flagged": 0, "n_cross_contaminated": 0},
                        "axis_C": {"n_ccg_duplicate_pairs": None, "max_units_per_tetrode": 210}},
    }
    table = compare_scoreboards(results)
    assert "variant" in table and "purB(sw)" in table
    assert "MP" in table and "chunk+match" in table
    assert "99.5" in table          # chunk+match coverage
    assert "0.910" in table         # MP spike-weighted purity formatted
    lines = table.splitlines()
    assert len(lines) == 3          # header + 2 variants

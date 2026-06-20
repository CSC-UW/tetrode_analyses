"""Unit tests for axis-B (assignment-purity) machinery.

Covers the pieces the reorientation adds so per-unit assignment correctness can be SCORED, not just
tetrode-pooled coverage + median rp_contamination (see MATCHING_PURSUIT_FINDINGS.md + the reorientation
plan):

  * ``_mp_common.all_template_cosines`` -- per-spike cosine to EVERY same-tetrode template; the best-match
    arg must pick the template the snippet actually resembles, not the (wrongly) assigned one.
  * ``_mp_common.competitive_reassign`` -- re-labels a spike to its best same-tetrode template instead of
    deleting it (the alternative to the delete-only shape_gate_r).
  * ``_assignment_eval.per_unit_best_match_purity`` -- aggregates per-spike best-match into a per-unit
    purity fraction + top contaminating neighbour.
  * ``_assignment_eval`` CCG arbiter (``adjudicate`` / ``verdict_of`` / ``ccg_verdict_pair``) -- the
    temporal duplicate-vs-distinct test promoted verbatim from script 63 (regression parity).
  * ``_assignment_eval.assignment_purity_summary`` -- the >= 2-signal impurity flag + category.

These modules live in the analyses script dir, not the installed package, so it is added to sys.path.
"""
import pathlib
import sys

import numpy as np
import pytest

_SORTING_DIR = (pathlib.Path(__file__).resolve().parents[1]
                / "analyses" / "tetrode_preprocessing_and_sorting" / "sorting")
sys.path.insert(0, str(_SORTING_DIR))

import spikeinterface as si  # noqa: E402  (after sys.path setup)

from _assignment_eval import (  # noqa: E402
    accumulate_best_match,
    adjudicate,
    assignment_purity_summary,
    ccg_verdict_pair,
    finalize_best_match,
    per_unit_best_match_purity,
    verdict_of,
)
from _mp_common import (  # noqa: E402
    all_template_cosines,
    asym_window_bounds,
    competitive_reassign,
    templates_from_dense,
)

FS = 30000.0
N_SAMP, NBEFORE, N_CHAN = 40, 15, 4


def _trough(amp, peak_ch, spread_ch):
    """A biphasic-ish trough on ``peak_ch`` (+ a little spread on ``spread_ch``), shape (N_SAMP, N_CHAN)."""
    t = np.arange(N_SAMP) - NBEFORE
    w = amp * np.exp(-(t**2) / 8.0)
    out = np.zeros((N_SAMP, N_CHAN), dtype=np.float32)
    out[:, peak_ch] = w
    out[:, spread_ch] = 0.3 * w
    return out


def _bank2():
    """Two-unit, single-tetrode bank with orthogonal footprints (u0 on ch0/1, u1 on ch2/3)."""
    t0 = _trough(-100.0, 0, 1)
    t1 = _trough(-100.0, 2, 3)
    dense = np.stack([t0, t1]).astype(np.float32)
    mask = np.ones((2, N_CHAN), dtype=bool)
    bank = templates_from_dense(dense, mask, NBEFORE, unit_ids=np.array([0, 1]),
                                channel_ids=np.arange(N_CHAN))
    return bank, dense


def _recording_with(placements):
    """NumpyRecording (4 ch, one tetrode) with (peak_sample, snippet) placements; snippet sits at
    tr[peak-NBEFORE : peak-NBEFORE+N_SAMP]."""
    total = max(p for p, _ in placements) + N_SAMP + 60
    traces = np.zeros((total, N_CHAN), dtype=np.float32)
    for peak, snip in placements:
        off = peak - NBEFORE
        traces[off:off + N_SAMP, :] = snip
    rec = si.NumpyRecording([traces], sampling_frequency=FS)
    rec.set_property("group", np.zeros(N_CHAN, dtype=int))
    return rec


def _spikes(samples, clusters):
    arr = np.zeros(len(samples), dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    arr["sample_index"] = samples
    arr["cluster_index"] = clusters
    return arr


# ---- all_template_cosines: best-match picks the resembled template -----------------------------

def test_all_template_cosines_best_match_is_resembled_unit():
    bank, dense = _bank2()
    pa, pb = 500, 900
    # pa: a u0-shaped snippet (correctly assigned to u0); pb: a u1-shaped snippet WRONGLY assigned to u0
    rec = _recording_with([(pa, dense[0]), (pb, dense[1])])
    spikes = _spikes([pa, pb], [0, 0])
    a, b = asym_window_bounds(NBEFORE)
    cos = all_template_cosines(rec, bank, spikes, a, b, s_max=2)

    assert cos["rF_u"][0] == pytest.approx(1.0)        # snippet == assigned u0 template
    assert cos["rF_arg"][0] == 0                       # u0 is its own best match
    assert cos["rF_u"][1] == pytest.approx(0.0, abs=1e-5)  # u1 snippet is orthogonal to u0
    assert cos["rF_best"][1] == pytest.approx(1.0)     # but it matches u1 perfectly
    assert cos["rF_arg"][1] == 1                       # so u1 is the best same-tetrode match


def test_all_template_cosines_out_of_bounds_is_arg_minus_one():
    bank, dense = _bank2()
    rec = _recording_with([(500, dense[0])])
    oob = rec.get_num_frames() + 5
    cos = all_template_cosines(rec, bank, _spikes([oob], [0]), *asym_window_bounds(NBEFORE), s_max=2)
    assert cos["rF_arg"][0] == -1
    assert np.isnan(cos["rF_best"][0])


# ---- competitive_reassign: relabel, not delete -------------------------------------------------

def test_competitive_reassign_relabels_wrong_assignment():
    bank, dense = _bank2()
    pa, pb = 500, 900
    rec = _recording_with([(pa, dense[0]), (pb, dense[1])])
    spikes = _spikes([pa, pb], [0, 0])  # both assigned to u0; pb really is u1
    out, reassigned = competitive_reassign(rec, bank, spikes, margin=0.0)

    assert out["cluster_index"][0] == 0       # u0-shaped spike stays on u0
    assert out["cluster_index"][1] == 1       # u1-shaped spike is moved to u1 (not deleted)
    assert reassigned.tolist() == [False, True]
    assert out["sample_index"].tolist() == [pa, pb]  # sample times untouched


def test_competitive_reassign_precomputed_cosines_matches():
    bank, dense = _bank2()
    pa, pb = 500, 900
    rec = _recording_with([(pa, dense[0]), (pb, dense[1])])
    spikes = _spikes([pa, pb], [0, 0])
    a, b = asym_window_bounds(NBEFORE)
    cos = all_template_cosines(rec, bank, spikes, a, b, s_max=2)
    # passing precomputed cosines (win/bank=None) reproduces the win/bank path
    out_a, re_a = competitive_reassign(rec, bank, spikes, margin=0.0)
    out_b, re_b = competitive_reassign(None, None, spikes, margin=0.0, cosines=cos)
    assert out_a["cluster_index"].tolist() == out_b["cluster_index"].tolist()
    assert re_a.tolist() == re_b.tolist()


def test_competitive_reassign_margin_blocks_marginal_moves():
    bank, dense = _bank2()
    pb = 900
    rec = _recording_with([(pb, dense[1])])
    spikes = _spikes([pb], [0])
    # margin just above the (perfect-match=1.0) - (orthogonal=0.0) gap would block; a 1.5 margin is
    # unreachable, so no reassignment despite u1 clearly winning.
    _, reassigned = competitive_reassign(rec, bank, spikes, margin=1.5)
    assert reassigned.tolist() == [False]


# ---- per_unit_best_match_purity ----------------------------------------------------------------

def test_per_unit_best_match_purity_counts_and_top_neighbor():
    # u0 (bank id 10): 10 finite spikes, 7 best=self, 3 best=u1; plus one out-of-bounds (arg=-1, excluded)
    ci = np.array([0] * 11 + [1] * 4, dtype=np.int64)
    rF_arg = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1] + [1, 1, 1, 1], dtype=np.int64)
    cosines = {"rF_arg": rF_arg, "rA_arg": rF_arg}
    spikes = _spikes(np.arange(ci.size) * 100 + 200, ci)
    out = per_unit_best_match_purity(spikes, cosines, bank_unit_ids=np.array([10, 20]))

    assert out[10]["n"] == 11
    assert out[10]["n_finite"] == 10                       # the -1 is dropped
    assert out[10]["best_match_frac"] == pytest.approx(0.7)
    assert out[10]["neighbor_win_frac"] == pytest.approx(0.3)
    assert out[10]["top_neighbor"] == 20                   # bank id of the contaminating neighbour
    assert out[10]["top_neighbor_frac"] == pytest.approx(0.3)
    assert out[20]["best_match_frac"] == pytest.approx(1.0)  # u1's spikes all best-match u1
    assert out[20]["top_neighbor"] == -1


def test_accumulate_finalize_matches_single_window():
    # one window through accumulate_best_match + finalize_best_match == per_unit_best_match_purity
    ci = np.array([0] * 11 + [1] * 4, dtype=np.int64)
    rF_arg = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1] + [1, 1, 1, 1], dtype=np.int64)
    cosines = {"rF_arg": rF_arg, "rA_arg": rF_arg}
    spikes = _spikes(np.arange(ci.size) * 100 + 200, ci)
    bank_ids = np.array([10, 20])
    direct = per_unit_best_match_purity(spikes, cosines, bank_ids)
    agg = finalize_best_match(accumulate_best_match({}, spikes, cosines, bank_ids))
    for uid in (10, 20):
        for k in ("n", "n_finite", "best_match_frac", "neighbor_win_frac", "top_neighbor", "top_neighbor_frac"):
            a, d = agg[uid][k], direct[uid][k]
            assert a == pytest.approx(d, nan_ok=True) if isinstance(d, float) else a == d


# ---- CCG arbiter (promoted from script 63): parity on the verdict map ---------------------------

@pytest.mark.parametrize("ratio,n_co,expected", [
    (0.1, 5, "duplicate"),       # deep dip -> same cell
    (1.0, 5, "distinct"),        # filled -> independent cells
    (0.5, 5, "ambiguous"),       # between thresholds
    (0.1, 1, "SEGREGATED"),      # too few co-active windows
    (np.nan, 5, "SEGREGATED"),   # too few flank coincidences -> ratio NaN
])
def test_verdict_of_thresholds(ratio, n_co, expected):
    assert verdict_of(ratio, n_co) == expected


def test_adjudicate_refractory_dip_low_ratio():
    # train B = train A shifted +300 frames (10 ms, in the flank band): never coincident at 0 lag
    a = np.sort(np.arange(200) * 120 + 1000).astype(np.int64)
    b = a + 300
    ratio, central, flank = adjudicate(b, a)
    assert central == 0          # no near-zero-lag coincidences
    assert flank > 30
    assert ratio == pytest.approx(0.0)


def test_ccg_verdict_pair_duplicate_vs_distinct():
    win = 10000
    # three co-active windows, >=5 spikes each
    base = np.concatenate([np.arange(40) * 200 + w * win + 500 for w in range(3)]).astype(np.int64)
    dup = base + 300                                  # 10 ms hand-off -> refractory dip -> duplicate
    rng = np.random.default_rng(0)
    indep = np.sort(np.concatenate(
        [rng.integers(w * win + 200, w * win + win - 200, size=120) for w in range(3)])).astype(np.int64)

    assert ccg_verdict_pair(base, dup, win_frames=win)["verdict"] == "duplicate"
    assert ccg_verdict_pair(base, indep, win_frames=win)["verdict"] == "distinct"


# ---- assignment_purity_summary: the >= 2-signal flag + category --------------------------------

def test_assignment_purity_summary_flag_and_categories():
    cosine_purity = {
        1: dict(best_match_frac=0.50, top_neighbor=2),   # impure cosine, neighbour=2 (distinct) -> flag
        3: dict(best_match_frac=0.95, top_neighbor=-1),  # clean
        4: dict(best_match_frac=0.50, top_neighbor=5),   # impure cosine, neighbour=5 duplicate
        6: dict(best_match_frac=0.50, top_neighbor=7),   # impure cosine, neighbour=7 duplicate
    }
    ccg = {
        (1, 2): dict(verdict="distinct"),
        (4, 5): dict(verdict="duplicate"),
        (6, 7): dict(verdict="duplicate"),
    }
    heldout = {
        1: dict(self_frac=0.50),   # impure -> u1 has cosine+ccg+heldout all impure
        3: dict(self_frac=0.95),
        4: dict(self_frac=0.95),   # u4: cosine impure(1) + ccg duplicate(not) + heldout clean -> 1 vote
        6: dict(self_frac=0.50),   # u6: cosine impure(1) + ccg duplicate(not) + heldout impure(1) -> 2
    }
    out = assignment_purity_summary(cosine_purity, ccg, heldout)

    assert out[1]["flagged"] is True and out[1]["category"] == "cross_contaminated"
    assert out[3]["flagged"] is False and out[3]["category"] == "clean"
    assert out[4]["flagged"] is False                      # only 1 impure vote (ccg duplicate is not impure)
    assert out[6]["flagged"] is True and out[6]["category"] == "oversplit"

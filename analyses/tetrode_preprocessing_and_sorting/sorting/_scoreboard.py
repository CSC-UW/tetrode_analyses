"""Unified three-axis scoreboard for tetrode sortings (head-to-head, no metric re-implemented).

Replaces the cherry-picked single-number tables in MATCHING_PURSUIT_FINDINGS.md with ONE comparison on
the three axes the goal actually has (see _assignment_eval + the reorientation plan). The headline
"wobble Pareto-dominates" rested on axis A (pooled) + median rp alone; this scores all three so the
chunk+match (2204 oversplit units, ~99.5% coverage) vs MP (~95 units) trade-off is surfaced, not assumed:

  A  event coverage   -- per-TETRODE coverage of detected high-MAD events, by amplitude band.
                         Delegates to _wobble_eval.coverage_by_band. Necessary, NOT sufficient (blind to
                         which unit on the tetrode claimed the event).
  B  assignment purity -- per-UNIT fraction assigned to the CORRECT same-tetrode unit, triangulated over
                         three internal signals. Delegates to _assignment_eval. This is the axis the
                         later work lacked.
  C  identity stability -- un-merged same-cell tracks (CCG 'duplicate' pairs) + unit-count parsimony.
                         Delegates to the _assignment_eval CCG arbiter.

The expensive inputs (detect_peaks for A; all_template_cosines for B) are computed ONCE by the caller
(scripts 86/89) and passed in, so every variant is scored on the SAME peaks/recording -- the
reuse-the-saved-events discipline of script 66.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from _assignment_eval import (accumulate_best_match, assignment_purity_summary, ccg_cross_contamination,
                              finalize_best_match, per_unit_best_match_purity, window_assignment_cosines)
from _mp_common import FS
from _wobble_eval import coverage_by_band


def windowed_axis_b(recording, sorting, windows_h, *, win_s=1800.0, n_jobs=16, fs=FS,
                    min_spikes_template=100):
    """Accumulate per-unit assignment-purity cosines over ``windows_h`` (full rF + tight rA), scoring each
    spike against window-local templates. Returns (cosine_purity_full, cosine_purity_tight). Shared by the
    B+C re-score (script 86) and the unified driver (script 89); each scores its sorting on its OWN
    recording. all_template_cosines is window-scale, so this never holds 48 h of trace in RAM.

    ``min_spikes_template`` = the per-window TEMPLATE-RELIABILITY floor; default 100 matches _mp_common's
    carry-forward ``min_spikes_reestimate=100`` (a 50-spike/1800 s template is noisy). A unit needs this
    many spikes in a window to get a window-local template + be scored there. Script 93 confirmed the
    spike-weighted purity is floor-INVARIANT to 3 decimals over {50,100,200} and the variant ranking is
    stable (retention drops <=1 unit 50->100), so the earlier floor-50 numbers (scripts 86/89/90) stand;
    100 is the SPOT-consistent pick.
    """
    nfr = recording.get_num_frames()
    acc_full: dict = {}
    acc_tight: dict = {}
    for h in windows_h:
        a = int(h * 3600 * fs)
        b = min(a + int(win_s * fs), nfr)
        if a >= nfr:
            continue
        spikes, cosines, bank_ids = window_assignment_cosines(recording, sorting, a, b, n_jobs=n_jobs,
                                                              min_spikes_template=min_spikes_template)
        if spikes is None:
            continue
        accumulate_best_match(acc_full, spikes, cosines, bank_ids, use_tight=False)
        accumulate_best_match(acc_tight, spikes, cosines, bank_ids, use_tight=True)
    return finalize_best_match(acc_full), finalize_best_match(acc_tight)


def axis_a_summary(sorting, peaks):
    """Axis A: per-tetrode coverage of detected peaks by MAD band. ``peaks`` = (peak_s, peak_g, amp_mad)."""
    peak_s, peak_g, amp_mad = peaks
    cov, _ = coverage_by_band(sorting, peak_s, peak_g, amp_mad)
    return cov


def axis_b_aggregate(cosine_purity, sorting, *, win_s=1800.0, heldout=None, **thresh):
    """Axis B from a PRECOMPUTED per-unit cosine purity (single-window ``per_unit_best_match_purity`` or
    cross-window ``finalize_best_match``). Runs the CCG arbiter ONLY on each unit's top cosine neighbour
    (signal 2, 'cosine proposes / CCG disposes'), folds in optional held-out agreement (signal 3), applies
    the >= 2-signal flag, and aggregates to spike-weighted purity + flagged / cross-contaminated /
    oversplit counts. ``**thresh`` -> assignment_purity_summary knobs.
    """
    pairs = [(uid, c["top_neighbor"]) for uid, c in cosine_purity.items() if c["top_neighbor"] >= 0]
    ccg = ccg_cross_contamination(sorting, pairs=pairs, win_s=win_s)
    per_unit = assignment_purity_summary(cosine_purity, ccg, heldout, **thresh)
    fracs = [c["best_match_frac"] for c in cosine_purity.values() if np.isfinite(c["best_match_frac"])]
    num = sum(c["n_finite"] * c["best_match_frac"] for c in cosine_purity.values()
              if np.isfinite(c["best_match_frac"]))
    den = sum(c["n_finite"] for c in cosine_purity.values() if np.isfinite(c["best_match_frac"]))
    n_units = len(cosine_purity)
    n_flag = int(sum(s["flagged"] for s in per_unit.values()))
    return {
        "n_units": n_units,
        "n_flagged": n_flag,
        "frac_flagged": float(n_flag / n_units) if n_units else float("nan"),
        "n_cross_contaminated": int(sum(s["category"] == "cross_contaminated" for s in per_unit.values())),
        "n_oversplit": int(sum(s["category"] == "oversplit" for s in per_unit.values())),
        "median_best_match_frac": float(np.median(fracs)) if fracs else float("nan"),
        "spikeweighted_purity": float(num / den) if den else float("nan"),
        "per_unit": per_unit,
        "cosine_purity": cosine_purity,
    }


def axis_b_summary(spikes, cosines, bank_unit_ids, sorting, *, win_s=1800.0, heldout=None,
                   use_tight=False, **thresh):
    """Axis B for a SINGLE window: compute per-unit cosine purity (signal 1) from ``spikes`` + ``cosines``
    (= all_template_cosines output), then delegate to ``axis_b_aggregate``. Multi-window callers (script
    86) accumulate purity across windows themselves and call ``axis_b_aggregate`` directly.
    """
    cosine_purity = per_unit_best_match_purity(spikes, cosines, bank_unit_ids, use_tight=use_tight)
    return axis_b_aggregate(cosine_purity, sorting, win_s=win_s, heldout=heldout, **thresh)


def axis_c_summary(sorting, *, win_s=1800.0, pairs=None, max_units_for_full_ccg=400):
    """Axis C: un-merged same-cell tracks + parsimony. CCG 'duplicate' pairs = same cell split across
    two un-merged tracks (an identity/merge failure). If ``pairs`` is None and the sorting has more than
    ``max_units_for_full_ccg`` units, the all-same-tetrode-pairs CCG is SKIPPED (O(units/tet^2) blow-up,
    e.g. chunk+match's 2204 units) -- pass candidate ``pairs`` from a template-cosine prefilter instead;
    the skip is reported (n_ccg_duplicate_pairs=None), never silent.
    """
    groups = np.asarray(sorting.get_property("group"))
    n_units = int(sorting.get_num_units())
    units_per_tet = Counter(int(g) for g in groups)
    out = {
        "n_units": n_units,
        "max_units_per_tetrode": int(max(units_per_tet.values())) if units_per_tet else 0,
        "mean_units_per_tetrode": float(np.mean(list(units_per_tet.values()))) if units_per_tet else 0.0,
    }
    if pairs is None and n_units > max_units_for_full_ccg:
        out.update(n_same_tet_pairs=None, n_ccg_duplicate_pairs=None, n_ccg_distinct_pairs=None,
                   ccg_skipped=True)
        return out
    ccg = ccg_cross_contamination(sorting, pairs=pairs, win_s=win_s)
    out.update(
        n_same_tet_pairs=len(ccg),
        n_ccg_duplicate_pairs=int(sum(v["verdict"] == "duplicate" for v in ccg.values())),
        n_ccg_distinct_pairs=int(sum(v["verdict"] == "distinct" for v in ccg.values())),
        ccg_skipped=False,
    )
    return out


def score_sorting(sorting, *, peaks=None, spikes=None, cosines=None, bank_unit_ids=None,
                  heldout=None, win_s=1800.0, use_tight=False, axis_c_pairs=None, label=None, **thresh):
    """Score a sorting on whichever axes the provided inputs allow. axis A needs ``peaks``; axis B needs
    ``spikes`` + ``cosines`` + ``bank_unit_ids``; axis C always runs (spike trains only).
    """
    result = {"label": label, "n_units": int(sorting.get_num_units())}
    if peaks is not None:
        result["axis_A"] = axis_a_summary(sorting, peaks)
    if spikes is not None and cosines is not None and bank_unit_ids is not None:
        result["axis_B"] = axis_b_summary(spikes, cosines, bank_unit_ids, sorting, win_s=win_s,
                                          heldout=heldout, use_tight=use_tight, **thresh)
    result["axis_C"] = axis_c_summary(sorting, win_s=win_s, pairs=axis_c_pairs)
    return result


_COLUMNS = [
    ("variant", "label", None),
    ("units", "n_units", None),
    ("covA>=10", ("axis_A", ">=10_pooled"), "{:.1f}"),
    ("covA>=12", ("axis_A", ">=12_pooled"), "{:.1f}"),
    ("purB(sw)", ("axis_B", "spikeweighted_purity"), "{:.3f}"),
    ("purB_t(sw)", ("axis_B", "spikeweighted_purity_tight"), "{:.3f}"),
    ("B_flag", ("axis_B", "n_flagged"), None),
    ("B_xcontam", ("axis_B", "n_cross_contaminated"), None),
    ("C_dup", ("axis_C", "n_ccg_duplicate_pairs"), None),
    ("C_max/tet", ("axis_C", "max_units_per_tetrode"), None),
]


def _dig(result, key):
    if key is None:
        return None
    if isinstance(key, tuple):
        cur = result
        for k in key:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        return cur
    return result.get(key)


def compare_scoreboards(results):
    """``{variant_label: score_sorting(...)}`` -> a printable head-to-head table (str) of the headline
    scalar from each axis. Whole per-unit detail stays in the result dicts; this is the at-a-glance view
    that replaces the findings doc's single-number tables.
    """
    header = [c[0] for c in _COLUMNS]
    rows = [header]
    for label, res in results.items():
        res = dict(res, label=res.get("label") or label)
        row = []
        for _, key, fmt in _COLUMNS:
            val = _dig(res, key)
            if val is None:
                row.append("-")
            elif fmt is not None and isinstance(val, (int, float)) and np.isfinite(val):
                row.append(fmt.format(val))
            else:
                row.append(str(val))
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(c.rjust(widths[i]) for i, c in enumerate(r)) for r in rows)

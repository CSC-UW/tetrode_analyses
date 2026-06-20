"""Shared, matcher-agnostic scoring for the wobble-vs-circus head-to-head (scripts 70, 71).

All four comparison axes, computed on a deduped per-window matcher output (geometry-free):
  * coverage_by_band   -- >=12 MAD detection coverage (reuses the script-64 detector + claimed logic)
  * spurious_fraction  -- matcher spikes with NO detected peak within +/-0.5 ms (over-detection)
  * over_detection     -- N_total + dedup-removed frac + residual same-tetrode duplicate frac
  * quality_df / precision_summary -- rp_contamination (Llobet), the matched-precision metric

Everything is in WINDOW-LOCAL frames (caller does win.reset_times()). Both matchers share one bank so
unit ids and per-tetrode pools are directly comparable. The precision metric deliberately does NOT gate
on contamination (it gates on a spike-count floor only), so it is not circular with the isolation tiers.
"""
from __future__ import annotations

import numpy as np
import spikeinterface as si
from spikeinterface.core import ChannelSparsity, get_noise_levels
from spikeinterface.metrics.quality.misc_metrics import (_compute_rp_contamination_one_unit,
                                                         _compute_rp_violations_numba)
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from _mp_common import tsq_median  # noqa: F401  (re-export: now lives in _mp_common for production use)
from _track_eval import QM_METRICS, TIERS, isolation_tier_mask

FS = 30000.0
TOL_MS = 0.5
T_R = int(round(1.0 * FS * 1e-3))  # 1 ms refractory (matches SI rp_violation default), censored t_c = 0
MIN_SPK = 50
DETECT_THRESH = 5.5
# 10-MAD boundary added so >=10 MAD coverage (the task-1 spec) is a reported band, not just >=12.
AMP_BINS = [5.5, 7, 9, 10, 12, 16, 24, np.inf]


def detect_window_peaks(win, *, detect_thresh=DETECT_THRESH, n_jobs=16):
    """(peak_s, peak_g, amp_mad) for a window: locally-exclusive neg peaks at 5.5 MAD (script-64 recipe)."""
    rec_groups = np.asarray(win.get_property("group"))
    noise = get_noise_levels(win, return_in_uV=False)
    peaks = detect_peaks(win, method="locally_exclusive", peak_sign="neg",
                         detect_threshold=detect_thresh, radius_um=40.0, noise_levels=noise,
                         n_jobs=n_jobs, chunk_duration="1s", progress_bar=False)
    peak_s = peaks["sample_index"].astype(np.int64)
    peak_ch = peaks["channel_index"].astype(np.int64)
    peak_g = rec_groups[peak_ch].astype(np.int64)
    amp_mad = np.abs(peaks["amplitude"]) / noise[peak_ch]
    return peak_s, peak_g, amp_mad


def _by_tet(sorting):
    """{tetrode group: sorted np.int64 spike frames pooled over that tetrode's units}."""
    grp = np.asarray(sorting.get_property("group"))
    out: dict = {}
    for i, u in enumerate(sorting.unit_ids):
        out.setdefault(int(grp[i]), []).append(sorting.get_unit_spike_train(u).astype(np.int64))
    return {g: np.sort(np.concatenate(v)) for g, v in out.items()}


def _within_tol(query_s, query_g, ref_by_tet, tol):
    """Bool per query event: a ref event within +/-tol on the SAME tetrode group."""
    out = np.zeros(len(query_s), dtype=bool)
    for g in np.unique(query_g):
        ref = ref_by_tet.get(int(g))
        sel = query_g == g
        if ref is None or ref.size == 0:
            continue
        qs = query_s[sel]
        j = np.searchsorted(ref, qs)
        dprev = np.where(j > 0, qs - ref[np.clip(j - 1, 0, ref.size - 1)], tol + 1)
        dnext = np.where(j < ref.size, ref[np.clip(j, 0, ref.size - 1)] - qs, tol + 1)
        out[np.flatnonzero(sel)] = np.minimum(dprev, dnext) <= tol
    return out


def coverage_by_band(sorting, peak_s, peak_g, amp_mad, *, tol=None):
    """({band: %claimed, '>=12_pooled': %, 'overall': %}, claimed_mask) for a sorting vs detected peaks.

    AXIS-A (event coverage) only, and POOLED PER TETRODE (see _by_tet): an event is "claimed" if ANY unit
    on its tetrode fired within tol, NOT if the CORRECT unit did. Necessary but NOT sufficient -- blind to
    per-unit assignment purity (axis B). For that use _assignment_eval (per_unit_best_match_purity)."""
    tol = int(TOL_MS * 1e-3 * FS) if tol is None else tol
    claimed = _within_tol(peak_s, peak_g, _by_tet(sorting), tol)
    res = {}
    for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
        m = (amp_mad >= lo) & (amp_mad < hi)
        band = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        res[band] = float(claimed[m].mean() * 100) if m.any() else float("nan")
    for lo in (10, 12):  # >=10 is the headline (task-1 spec); >=12 kept for continuity
        big = amp_mad >= lo
        res[f">={lo}_pooled"] = float(claimed[big].mean() * 100) if big.any() else float("nan")
    res["overall"] = float(claimed.mean() * 100)
    return res, claimed


def spurious_fraction(sorting, peak_s, peak_g, *, tol=None):
    """Fraction of MATCHER spikes with NO detected peak within +/-tol on the same tetrode (over-detection)."""
    tol = int(TOL_MS * 1e-3 * FS) if tol is None else tol
    pbt = {int(g): np.sort(peak_s[peak_g == g]) for g in np.unique(peak_g)}
    grp = np.asarray(sorting.get_property("group"))
    qs, qg = [], []
    for i, u in enumerate(sorting.unit_ids):
        tr = sorting.get_unit_spike_train(u).astype(np.int64)
        qs.append(tr)
        qg.append(np.full(tr.size, int(grp[i]), np.int64))
    if not qs or sum(t.size for t in qs) == 0:
        return float("nan")
    near = _within_tol(np.concatenate(qs), np.concatenate(qg), pbt, tol)
    return float(1.0 - near.mean())


def over_detection(sorting, n_pre_dedup, *, coincidence_ms=0.5):
    """N_total + dedup-removed frac + residual same-tetrode duplicate frac (coincidences within coincidence_ms)."""
    tol = int(coincidence_ms * 1e-3 * FS)
    sbt = _by_tet(sorting)
    n_total = int(sum(t.size for t in sbt.values()))
    n_uniq = 0
    for t in sbt.values():
        n_uniq += int(1 + np.sum(np.diff(t) > tol)) if t.size > 1 else int(t.size)
    return {
        "n_total": n_total,
        "n_pre_dedup": int(n_pre_dedup),
        "dedup_removed_frac": float(1 - n_total / n_pre_dedup) if n_pre_dedup else float("nan"),
        "residual_dup_frac": float(1 - n_uniq / n_total) if n_total else float("nan"),
    }


def quality_df(sorting, recording, *, n_jobs=16, with_amplitude_cutoff=False):
    """qm_df with QM_METRICS (rp_contamination, firing_rate, isi_violation) for a matcher output (raw units).

    with_amplitude_cutoff=True also computes BombCell's recall-side metric `amplitude_cutoff` (estimated
    fraction of a unit's spikes MISSED below detection threshold, from the amplitude-distribution
    truncation; needs the spike_amplitudes extension). It is the false-NEGATIVE complement to
    rp_contamination's false-positive: as the matcher threshold drops and low-amplitude spikes are
    recovered, amplitude_cutoff should fall while rp_contamination rises -> a two-sided knee.
    """
    sparsity = ChannelSparsity.from_property(sorting, recording, by_property="group")
    az = si.create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity, return_in_uV=False)
    az.compute({"random_spikes": {"seed": 0}, "noise_levels": {}, "waveforms": {}, "templates": {}}, n_jobs=n_jobs)
    metric_names = list(QM_METRICS)
    if with_amplitude_cutoff:
        az.compute({"spike_amplitudes": {}}, n_jobs=n_jobs)
        metric_names.append("amplitude_cutoff")
    az.compute({"quality_metrics": {"metric_names": metric_names}})
    return az.get_extension("quality_metrics").get_data()


def precision_summary(sorting, qm_df, *, min_spk=50):
    """Precision PROXY (rp_contamination), NOT assignment purity. Median rp_contamination over units with
    >=min_spk spikes (RATE-based inclusion only, NOT gated on contamination -> not circular with the tiers);
    also per-tier well-isolated unit counts. rp_contamination is BLIND to cross-unit mis-assignment: an
    independently-firing same-tetrode neighbour wrongly assigned into a unit raises no refractory violation,
    so rp~0 is not evidence of correct assignment. For per-unit assignment purity (axis B) use
    _assignment_eval (per_unit_best_match_purity + the CCG arbiter)."""
    counts = {int(u): int(sorting.get_unit_spike_train(u).size) for u in sorting.unit_ids}
    rp = np.asarray(qm_df["rp_contamination"], dtype=float)
    ids = list(qm_df.index)
    keep = np.array([counts.get(int(i), 0) >= min_spk for i in ids], dtype=bool)
    rp_keep = rp[keep]
    rp_keep = rp_keep[np.isfinite(rp_keep)]
    return {
        "median_rp_contamination": float(np.median(rp_keep)) if rp_keep.size else float("nan"),
        "mean_rp_contamination": float(np.mean(rp_keep)) if rp_keep.size else float("nan"),
        "n_units_ge_min_spk": int(keep.sum()),
        "n_rp_estimable": int(rp_keep.size),
        "n_units": int(len(ids)),
        "tier_counts": {t: int(isolation_tier_mask(qm_df, t).sum()) for t in TIERS},
    }


def rp_contam(samples, total_samples, *, t_r=T_R):
    """Llobet rp_contamination for one spike train (SI internals; censored t_c=0). NaN if <2 spikes.

    The fast, analyzer-free precision primitive for the gate sweeps (scripts 79/81): scoring N gate
    settings per window via create_sorting_analyzer would be prohibitive, so contamination is computed
    directly on per-unit spike trains.
    """
    if samples.size < 2:
        return float("nan")
    nv = _compute_rp_violations_numba(np.sort(samples).astype(np.int64), 0, t_r)
    return _compute_rp_contamination_one_unit(nv, int(samples.size), int(total_samples), 0, t_r)


def score_kept_spikes(s, ci, ug, peak_s, peak_g, amp_mad, peak_by_tet, nfr, *, min_spk=MIN_SPK):
    """Analyzer-free score of a KEPT spike set (samples s, cluster indices ci) vs the window's peaks.

    Returns median rp (Llobet, over >=min_spk units), >=10 / >=12 MAD coverage, low-amp (5.5-10 MAD)
    retention, spurious fraction, and counts. ``ug`` = tetrode group per template index; ``peak_by_tet``
    = {group: sorted detected-peak samples}. Shared by the gate bake-off (79) and the admit x r grid (81).
    """
    tol = int(TOL_MS * 1e-3 * FS)
    g = ug[ci]
    by_unit = {int(u): np.sort(s[ci == u]) for u in np.unique(ci)}
    by_tet = {int(gg): np.sort(s[g == gg]) for gg in np.unique(g)}
    claimed = _within_tol(peak_s, peak_g, by_tet, tol)

    def cov(mask):
        return float(claimed[mask].mean() * 100) if mask.any() else float("nan")

    rps = [rp_contam(v, nfr) for v in by_unit.values() if v.size >= min_spk]
    rps = [v for v in rps if np.isfinite(v)]
    near = _within_tol(s, g, peak_by_tet, tol) if s.size else np.zeros(0, bool)
    return {
        "n_spikes": int(s.size), "n_units": int(len(by_unit)),
        "median_rp": float(np.median(rps)) if rps else float("nan"),
        "cov10": cov(amp_mad >= 10), "cov12": cov(amp_mad >= 12),
        "cov_low": cov((amp_mad >= 5.5) & (amp_mad < 10)),
        "spurious": float(1 - near.mean()) if s.size else float("nan"),
    }

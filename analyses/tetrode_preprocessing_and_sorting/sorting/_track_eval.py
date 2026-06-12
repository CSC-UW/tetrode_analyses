"""Shared helpers for the cross-chunk tracking experiments + curation tiering (SPOT).

The well-isolated tier definition (see ``isolation_tier_mask``) gates on a firing-rate
floor AND ``rp_contamination`` OR ``sliding_rp_violation``; thresholds are pulled from
``ecephys.wne.siutils`` so the definition lives in one place. ``isi_violations_ratio`` is
intentionally NOT gated (its ~1/rate^2 inflation over-rejects low-rate L2/3 units);
``presence_ratio`` was dropped (tracks scheme-3 block merging, not isolation);
``snr``/``amplitude_cutoff`` were non-discriminating. See TRACKING_FINDINGS.md (2026-06-12).
"""
from __future__ import annotations

import pathlib

import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
from spikeinterface.core import ChannelSparsity

from ecephys.wne import siutils

from tetrode_analyses import tracking as tk

TIERS = ("permissive", "moderate", "conservative")

# Quality metrics for the reference analyzer used in the validation scripts (cheap,
# spike-train based). rp_violation -> rp_contamination and firing_rate drive the gate;
# isi_violation is kept for reporting but is not gated. sliding_rp_violation is not
# computed here (a short reference epoch has too few coincidences), so the gate falls
# back to rp_contamination only -- see isolation_tier_mask.
QM_METRICS = ["firing_rate", "isi_violation", "rp_violation"]


def single_sort_reference(
    store_zarr,
    work_dir,
    *,
    t0_frame: int,
    t1_frame: int,
    fs: float,
    training_duration_sec: float = 3600.0,
    sort_n_jobs: int = 5,
    materialize_n_jobs: int = 96,
):
    """One MS5 scheme-2 sort of the whole span [t0,t1) as a single block, plus a
    quality-metrics analyzer. Returns ``(sorting, analyzer, qm_df, binary)``.

    The span must be drift-stable for this to be a fair "ground truth": a single
    classifier then captures every well-isolated unit, and any reconstruction
    disagreement is a tracking failure, not drift.
    """
    work_dir = pathlib.Path(work_dir)
    chunk = tk.Chunk(index=0, start_frame=int(t0_frame), end_frame=int(t1_frame), fs=fs)
    binary = tk.materialize_chunk(store_zarr, chunk, work_dir / "ref_bin", cmr="global", n_jobs=materialize_n_jobs)
    sorting = tk.sort_chunk(
        binary, work_dir / "ref_sort", scheme2_training_duration_sec=training_duration_sec, sort_n_jobs=sort_n_jobs
    )
    sorting = tk.to_int_numpy_sorting(sorting)  # int64 ids (compare_two_sorters rejects uint64)
    # analyzer needs the sorting on the crop's local (0-based) frame base
    sparsity = ChannelSparsity.from_property(sorting, binary, by_property="group")
    analyzer = si.create_sorting_analyzer(sorting, binary, format="memory", sparsity=sparsity, return_in_uV=True)
    analyzer.compute({"random_spikes": {}, "noise_levels": {}, "waveforms": {}, "templates": {}}, n_jobs=sort_n_jobs)
    analyzer.compute({"quality_metrics": {"metric_names": QM_METRICS}}, n_jobs=sort_n_jobs)
    qm_df = analyzer.get_extension("quality_metrics").get_data()
    # shift to absolute recording frames so it compares against the absolute-framed
    # assembled reconstruction (unit ids/group preserved -> qm_df still aligns)
    sorting_abs = tk.shift_sorting(sorting, int(t0_frame))
    return sorting_abs, analyzer, qm_df, binary


def isolation_tier_mask(qm_df, tier: str) -> np.ndarray:
    """Boolean array over ``qm_df`` rows: well-isolated at ``tier``.

    ``(rp_contamination <= rp_hi) OR (sliding_rp_violation <= srp_hi)`` AND
    ``firing_rate >= fr_lo``, thresholds from ``ecephys.wne.siutils``. A NaN in either
    refractory metric FAILS that metric's clause -- in particular a NaN
    ``sliding_rp_violation`` means "too few coincidences to evaluate" (abstain) and must
    never pass. If the ``sliding_rp_violation`` column is absent (a light reference
    analyzer), the gate falls back to ``rp_contamination`` only.
    """
    fr_lo = siutils.required_metric_thresholds["firing_rate"][tier][0]
    rp_hi = siutils.isolation_metric_thresholds["rp_contamination"][tier][1]
    fr = np.asarray(qm_df["firing_rate"], dtype=float)
    rp = np.asarray(qm_df["rp_contamination"], dtype=float)
    rp_ok = np.isfinite(rp) & (rp <= rp_hi)
    if "sliding_rp_violation" in qm_df.columns:
        srp_hi = siutils.isolation_metric_thresholds["sliding_rp_violation"][tier][1]
        srp = np.asarray(qm_df["sliding_rp_violation"], dtype=float)
        srp_ok = np.isfinite(srp) & (srp <= srp_hi)
    else:
        srp_ok = np.zeros(len(qm_df), dtype=bool)
    return (rp_ok | srp_ok) & (fr >= fr_lo)


def good_unit_ids(qm_df, present_ids, tier: str) -> np.ndarray:
    """Well-isolated reference unit ids for ``tier`` (intersected with ``present_ids``)."""
    ids = np.asarray(qm_df.index[isolation_tier_mask(qm_df, tier)])
    present = set(np.asarray(present_ids).tolist())
    return np.array([i for i in ids if i in present])


def score_reconstruction(ref_sorting, qm_df, recon_sorting, *, delta_time=0.4, match_score=0.5) -> dict:
    """Agreement of a reconstruction vs the single-sort reference, per tier.

    Returns ``{tier: {n_ref, matched, match_frac, mean_agreement}, ...}`` including
    an ``all_units`` row, plus ``n_recon`` and ``n_ref`` totals. The pass gate is
    well-isolated match_frac and mean_agreement both >= 0.9.
    """
    ref_ids = np.asarray(ref_sorting.unit_ids)
    cmp = sc.compare_two_sorters(
        ref_sorting, recon_sorting, sorting1_name="ref", sorting2_name="recon",
        delta_time=delta_time, match_score=match_score,
    )
    m = cmp.get_matching()[0]
    ag = cmp.agreement_scores
    out = {"n_ref_total": int(len(ref_ids)), "n_recon_total": int(recon_sorting.get_num_units())}
    labels = [("all_units", ref_ids)] + [(t, good_unit_ids(qm_df, ref_ids, t)) for t in TIERS]
    for label, ids in labels:
        ids = np.asarray(ids)
        matched = [u for u in ids if m.get(u, -1) != -1]
        frac = len(matched) / len(ids) if len(ids) else float("nan")
        mean_ag = float(np.nanmean([ag.loc[u, m[u]] for u in matched])) if matched else float("nan")
        out[label] = {
            "n_ref": int(len(ids)),
            "matched": int(len(matched)),
            "match_frac": round(frac, 3),
            "mean_agreement": round(mean_ag, 4),
        }
    return out

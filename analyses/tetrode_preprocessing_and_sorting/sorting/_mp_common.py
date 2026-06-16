"""Shared helpers for the matching-pursuit tracking experiments (scripts 51+).

Centralizes the hard-won correct logic discovered in the PoC (see project memory
`project_tetrode_matching_pursuit`):

  * UNITS MUST MATCH: circus-omp reads the recording in RAW units; the materialized binary
    carries gain_to_uV=0.195, so templates must be built in raw units too (analyzer
    return_in_uV=False + get_dense_templates_array(return_in_uV=False) + Templates is_in_uV=False),
    else templates are ~5x too small and OMP over-detects.
  * Templates with a sparsity_mask want the SPARSE array (n_units, n_samples, max_active=4),
    packed per unit onto its tetrode's channels.

Conventions mirror tracking.py (chunk-local frames; group sparsity per tetrode; int64 unit ids).
"""
from __future__ import annotations

import pathlib
import shutil

import numpy as np
import spikeinterface as si
from spikeinterface.core import ChannelSparsity, Templates
from spikeinterface.core.template_tools import get_dense_templates_array
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

from tetrode_analyses.tracking import (
    Chunk,
    _UnionFind,
    cosine_from_templates,
    materialize_chunk,
    sort_chunk,
    to_int_numpy_sorting,
)

FS = 30000.0
ZARR = pathlib.Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/2026-05-27_09-07-52.blosc-zstd.zarr"
)


def prepare_span(out_dir, start_s, dur_s, *, materialize_jobs=96):
    """Materialize bandpass+CMR + MS5-sort a span, caching to out_dir.

    Returns (recording, ref_int_sorting). On a second call with the same out_dir the cached
    binary + reference sort are reloaded (no re-materialize / re-sort).
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_dir, ref_dir = out_dir / "binary", out_dir / "ref_int"
    if bin_dir.exists() and ref_dir.exists():
        try:
            return si.load(bin_dir), si.load(ref_dir)
        except Exception:
            shutil.rmtree(ref_dir, ignore_errors=True)
    chunk = Chunk(index=0, start_frame=int(start_s * FS), end_frame=int((start_s + dur_s) * FS), fs=FS)
    rec = materialize_chunk(ZARR, chunk, bin_dir, cmr="global", n_jobs=materialize_jobs)
    shutil.rmtree(out_dir / "ref_sort", ignore_errors=True)
    ref = to_int_numpy_sorting(sort_chunk(rec, out_dir / "ref_sort"))
    shutil.rmtree(ref_dir, ignore_errors=True)
    ref.save(folder=ref_dir)
    return rec, ref


def materialize_span(out_dir, start_s, dur_s, *, materialize_jobs=96):
    """Materialize bandpass+CMR for a span (cached), WITHOUT sorting. Returns the recording."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = out_dir / "binary"
    if bin_dir.exists():
        try:
            return si.load(bin_dir)
        except Exception:
            shutil.rmtree(bin_dir, ignore_errors=True)
    chunk = Chunk(index=0, start_frame=int(start_s * FS), end_frame=int((start_s + dur_s) * FS), fs=FS)
    return materialize_chunk(ZARR, chunk, bin_dir, cmr="global", n_jobs=materialize_jobs)


def build_templates_object(sorting, recording, *, ms_before=1.0, ms_after=2.0, n_jobs=16, with_snr=True):
    """Group-sparse, RAW-unit Templates bank from a sorting+recording. Returns (templates, analyzer)."""
    sparsity = ChannelSparsity.from_property(sorting, recording, by_property="group")
    az = si.create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity, return_in_uV=False)
    az.compute({"random_spikes": {}, "waveforms": {"ms_before": ms_before, "ms_after": ms_after},
                "templates": {}, "noise_levels": {}}, n_jobs=n_jobs)
    if with_snr:
        az.compute({"quality_metrics": {"metric_names": ["snr", "firing_rate"]}})
    dense = get_dense_templates_array(az, return_in_uV=False)  # (n_units, n_samp, n_chan), raw units
    mask = az.sparsity.mask
    nbefore = az.get_extension("templates").nbefore
    n_units, n_samp, _ = dense.shape
    n_act = int(mask.sum(axis=1).max())
    sparse_arr = np.zeros((n_units, n_samp, n_act), dtype=np.float32)
    for i in range(n_units):
        ch = np.flatnonzero(mask[i])
        sparse_arr[i, :, : ch.size] = dense[i][:, ch]
    templates = Templates(
        templates_array=sparse_arr, sampling_frequency=FS, nbefore=nbefore, is_in_uV=False,
        sparsity_mask=mask, channel_ids=np.asarray(recording.channel_ids),
        unit_ids=np.asarray(sorting.unit_ids), probe=None, check_for_consistent_sparsity=True,
    )
    return templates, az


def _pack_sparse(dense, mask):
    """Dense (n_units, n_samp, n_chan) + bool mask -> sparse (n_units, n_samp, max_active)."""
    n_units, n_samp, _ = dense.shape
    n_act = int(mask.sum(axis=1).max())
    out = np.zeros((n_units, n_samp, n_act), dtype=np.float32)
    for i in range(n_units):
        ch = np.flatnonzero(mask[i])
        out[i, :, : ch.size] = dense[i][:, ch]
    return out


def templates_from_dense(dense, mask, nbefore, unit_ids, channel_ids):
    """Build a RAW-unit, group-sparse Templates from a dense template array + mask."""
    return Templates(
        templates_array=_pack_sparse(dense, mask), sampling_frequency=FS, nbefore=nbefore,
        is_in_uV=False, sparsity_mask=mask, channel_ids=np.asarray(channel_ids),
        unit_ids=np.asarray(unit_ids), probe=None, check_for_consistent_sparsity=True)


def _unit_groups_from_mask(mask, rec_groups):
    """Tetrode group of each unit = group of its (single-tetrode) active channels."""
    return np.array([int(rec_groups[np.flatnonzero(m)[0]]) for m in mask])


def windowed_carry_forward(recording, init_templates, *, window_s=900.0, method_kwargs=None,
                           n_jobs=16, reestimate=True, min_spikes_reestimate=15,
                           ms_before=1.0, ms_after=2.0, reestimate_min_cos=None):
    """Detect a fixed unit set across the recording window-by-window, carrying templates forward.

    Each window: run circus-omp with the current bank; if reestimate, re-derive each PRESENT unit's
    template from this window's detections (tracking drift) while KEEPING the prior template for units
    absent this window (so a unit that drops out for one window is still sought in the next -- the
    dropout-recovery mechanism). Returns (assembled NumpySorting over the full recording in absolute
    frames, counts array of shape (n_windows, n_units)).

    ``reestimate_min_cos`` (default None = off): per-window re-estimation STEP-CAP. When set (e.g. 0.8),
    a window's re-estimated template is ACCEPTED only if its shift-tolerant 4-channel cosine to the
    current template is >= this value; otherwise the update is REJECTED and the current template is kept
    (frozen for that window). Gradual drift (cos ~0.95/window) passes; an abrupt one-window jump --- the
    signature of a track being captured by a louder same-tetrode neighbor (see MATCHING_PURSUIT_FINDINGS
    "IDENTITY-SWAP") --- is blocked. Prevents re-estimation capture without harming real drift-tracking.
    """
    import spikeinterface.sortingcomponents.matching as _m  # noqa: F401 (ensure registered)
    fs = recording.get_sampling_frequency()
    total = recording.get_num_frames()
    wlen = int(window_s * fs)
    rec_groups = np.asarray(recording.get_property("group"))
    unit_ids = np.asarray(init_templates.unit_ids)
    mask = init_templates.sparsity.mask
    unit_groups = _unit_groups_from_mask(mask, rec_groups)
    nbefore = init_templates.nbefore
    cur_dense = np.asarray(init_templates.get_dense_templates(), dtype=np.float32)  # (n_units, n_samp, n_chan)

    bounds = [(s, min(s + wlen, total)) for s in range(0, total, wlen)]
    all_samples, all_labels, counts = [], [], []
    n_capped = 0  # re-estimation updates rejected by the step-cap (reestimate_min_cos)
    for a, b in bounds:
        win = recording.frame_slice(a, b)
        win.reset_times()
        templates = templates_from_dense(cur_dense, mask, nbefore, unit_ids, recording.channel_ids)
        mp, spikes = run_matching(win, templates, method_kwargs=method_kwargs, n_jobs=n_jobs)
        ufield = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
        s_idx = spikes["sample_index"].astype(np.int64)
        l_idx = spikes[ufield].astype(np.int64)
        all_samples.append(s_idx + a)
        all_labels.append(unit_ids[l_idx])
        cnt = np.bincount(l_idx, minlength=len(unit_ids))
        counts.append(cnt)
        if reestimate:
            present = np.flatnonzero(cnt >= min_spikes_reestimate)
            if present.size:
                ws = si.NumpySorting.from_samples_and_labels(
                    [s_idx], [unit_ids[l_idx]], sampling_frequency=fs, unit_ids=unit_ids)
                ws.set_property("group", unit_groups)
                ws = ws.select_units(unit_ids[present])
                new_t, _ = build_templates_object(ws, win, with_snr=False, n_jobs=n_jobs,
                                                  ms_before=ms_before, ms_after=ms_after)
                new_dense = np.asarray(new_t.get_dense_templates(), dtype=np.float32)
                for k, ui in enumerate(present):
                    if reestimate_min_cos is not None:
                        ch = np.flatnonzero(rec_groups == unit_groups[ui])  # unit's 4 tetrode channels
                        c = cosine_from_templates(cur_dense[ui][:, ch], new_dense[k][:, ch],
                                                  max_shift_samples=10)
                        if c < reestimate_min_cos:
                            n_capped += 1  # reject: abrupt jump -> keep current template (freeze)
                            continue
                    cur_dense[ui] = new_dense[k]
    if reestimate_min_cos is not None:
        print(f"  [step-cap reestimate_min_cos={reestimate_min_cos}] rejected {n_capped} "
              f"re-estimation updates (frozen) across {len(bounds)} windows", flush=True)
    samples = np.concatenate(all_samples)
    labels = np.concatenate(all_labels)
    assembled = si.NumpySorting.from_samples_and_labels(
        [samples], [labels], sampling_frequency=fs, unit_ids=unit_ids)
    assembled.set_property("group", unit_groups)
    return assembled, np.array(counts)


def run_matching(recording, templates, *, method="circus-omp", method_kwargs=None, n_jobs=16):
    """Run a matching-pursuit method; return (NumpySorting in template-unit ids, raw spikes array)."""
    spikes = find_spikes_from_templates(
        recording, templates, method=method, method_kwargs=method_kwargs or {},
        job_kwargs={"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": False})
    names = spikes.dtype.names
    uf = "cluster_index" if "cluster_index" in names else "unit_index"
    samples = spikes["sample_index"].astype(np.int64)
    labels = np.asarray(templates.unit_ids)[spikes[uf].astype(np.int64)]
    mp = si.NumpySorting.from_samples_and_labels(
        [samples], [labels], sampling_frequency=FS, unit_ids=np.asarray(templates.unit_ids))
    return mp, spikes


def dedup_sorting(sorting, recording, *, cosine_min=0.9, max_shift_samples=10, coincidence_ms=0.3):
    """Merge WITHIN-tetrode near-identical units (template cosine >= cosine_min) via union-find.

    Returns a merged NumpySorting (int64 ids, 'group' preserved); spike trains of merged units are
    unioned with coincident spikes (<coincidence_ms) collapsed. Geometry-free: cosine is over each
    unit's own 4-channel template (with small time-shift tolerance). This removes the MS5 oversplit
    redundancy that makes matching pursuit split a spike's assignment across near-identical twins.
    """
    templates, az = build_templates_object(sorting, recording, with_snr=False)
    dense = get_dense_templates_array(az, return_in_uV=False)
    mask = az.sparsity.mask
    uids = list(sorting.unit_ids)
    groups = np.asarray(sorting.get_property("group"))
    uid_to_idx = {u: i for i, u in enumerate(uids)}

    union = _UnionFind()
    for u in uids:
        union.add(u)
    by_group: dict = {}
    for u, g in zip(uids, groups):
        by_group.setdefault(int(g), []).append(u)
    for g, members in by_group.items():
        for a in range(len(members)):
            ia = uid_to_idx[members[a]]
            ch = np.flatnonzero(mask[ia])  # same group -> same 4 channels
            ta = dense[ia][:, ch]
            for b in range(a + 1, len(members)):
                ib = uid_to_idx[members[b]]
                tb = dense[ib][:, ch]
                if cosine_from_templates(ta, tb, max_shift_samples=max_shift_samples) >= cosine_min:
                    union.union(members[a], members[b])

    comp: dict = {}
    for u in uids:
        comp.setdefault(union.find(u), []).append(u)
    tol = int(coincidence_ms * 1e-3 * FS)
    trains, new_groups = {}, []
    for new_id, members in enumerate(comp.values()):
        allspk = np.sort(np.concatenate([sorting.get_unit_spike_train(u) for u in members]))
        if allspk.size and tol > 0:
            allspk = allspk[np.concatenate([[True], np.diff(allspk) > tol])]
        trains[new_id] = allspk
        new_groups.append(int(groups[uid_to_idx[members[0]]]))
    merged = si.NumpySorting.from_unit_dict([trains], sampling_frequency=FS)
    merged.set_property("group", np.asarray(new_groups))
    return merged


def detection_recall(ref_sorting, mp_spikes_samples, ref_unit_mask=None, *, tol_ms=0.5):
    """Label-agnostic recall: fraction of selected ref spikes within tol_ms of ANY matching spike."""
    uids = list(ref_sorting.unit_ids)
    sel = uids if ref_unit_mask is None else [u for u, m in zip(uids, ref_unit_mask) if m]
    if not sel:
        return float("nan")
    ref = np.sort(np.concatenate([ref_sorting.get_unit_spike_train(u) for u in sel]).astype(np.int64))
    s = np.sort(np.asarray(mp_spikes_samples, dtype=np.int64))
    if s.size == 0:
        return 0.0
    tol = int(tol_ms * 1e-3 * FS)
    j = np.searchsorted(s, ref)
    dprev = np.where(j > 0, ref - s[np.clip(j - 1, 0, len(s) - 1)], tol + 1)
    dnext = np.where(j < len(s), s[np.clip(j, 0, len(s) - 1)] - ref, tol + 1)
    return float(np.mean(np.minimum(dprev, dnext) <= tol))

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
from spikeinterface.core import ChannelSparsity, Templates, get_noise_levels
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


def build_templates_object(sorting, recording, *, ms_before=1.0, ms_after=2.0, n_jobs=16, with_snr=True, seed=0):
    """Group-sparse, RAW-unit Templates bank from a sorting+recording. Returns (templates, analyzer).

    ``seed`` (default 0) pins random_spikes so the templates -- and hence dedup_sorting's borderline
    merge decisions -- are REPRODUCIBLE run-to-run (otherwise unseeded sampling jitters the dedup'd
    seed-bank size, e.g. 73-77 units across nominally identical runs).
    """
    sparsity = ChannelSparsity.from_property(sorting, recording, by_property="group")
    az = si.create_sorting_analyzer(sorting, recording, format="memory", sparsity=sparsity, return_in_uV=False)
    az.compute({"random_spikes": {"seed": seed}, "waveforms": {"ms_before": ms_before, "ms_after": ms_after},
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


def tsq_median(bank):
    """Median per-template ||t||^2 (sum of squares over samples x active channels) of a sparse bank.

    Wobble's detection objective is ~||t||^2-scaled, so a per-window wobble threshold set as
    ``factor * tsq_median(bank)`` tracks each window's template energy (the adaptive-||t||^2 gate). It
    lives HERE (not in _wobble_eval) so the production carry-forward loops can recompute it per window
    without importing the eval harness; _wobble_eval re-exports it for the study scripts.
    """
    arr = np.asarray(bank.templates_array, dtype=np.float64)
    return float(np.median((arr ** 2).sum(axis=(1, 2))))


def windowed_carry_forward(recording, init_templates, *, window_s=900.0, method="circus-omp",
                           method_kwargs=None, shape_gate_r=None, wobble_factor=None,
                           n_jobs=16, reestimate=True, min_spikes_reestimate=100,
                           ms_before=1.0, ms_after=2.0, reestimate_min_cos=None, max_windows=None):
    """Detect a fixed unit set across the recording window-by-window, carrying templates forward.

    Each window: run the matcher (``method``, default "circus-omp"; "wobble" sets its per-window
    threshold from ``wobble_factor``, and ``shape_gate_r`` applies the scale-invariant cosine
    acceptance gate) with the current bank; if reestimate, re-derive each PRESENT unit's
    template from this window's detections (tracking drift) while KEEPING the prior template for units
    absent this window (so a unit that drops out for one window is still sought in the next -- the
    dropout-recovery mechanism). Returns (assembled NumpySorting over the full recording in absolute
    frames, counts array of shape (n_windows, n_units)).

    ``min_spikes_reestimate`` (default 100) is the TEMPLATE-RELIABILITY FLOOR: re-estimate a unit's
    template from a window only if it fired >= this many spikes there, else carry the prior (higher-
    confidence) template -- i.e. never downgrade a >=100-spike template with a sparser, noisier estimate.
    It matches the seed/re-seed admission bar (a re-estimated template needs as many spikes as the
    initial one); runner scripts tie it to their --min-spikes by default, and decouple it only when the
    admission bar is lowered to admit marginal/low-rate units (then carrying the trusted template beats
    re-estimating from too few -- drift here is mild).

    ``reestimate_min_cos`` (default None = off): per-window re-estimation STEP-CAP. When set (e.g. 0.8),
    a window's re-estimated template is ACCEPTED only if its shift-tolerant 4-channel cosine to the
    current template is >= this value; otherwise the update is REJECTED and the current template is kept
    (frozen for that window). Gradual drift (cos ~0.95/window) passes; an abrupt one-window jump --- the
    signature of a track being captured by a louder same-tetrode neighbor (see MATCHING_PURSUIT_FINDINGS
    "IDENTITY-SWAP") --- is blocked.

    MEASURED LIMITATION (2026-06-15): the observed swaps are GRADUAL multi-window walks (~0.99 cosine
    per window, indistinguishable from a stable unit's re-estimation noise), NOT abrupt jumps -- so this
    per-window cap is a no-op against them (a cap tight enough to catch the walk, ~0.99, also fires on
    stable tracks). Kept as a guard against TRUE one-window jumps; the gradual-capture fix is template
    COMPETITION (periodic re-seeding), not step-capping.
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
    if max_windows:
        bounds = bounds[:max_windows]  # parity with windowed_carry_forward_reseed (A/B + smoke knob)
    all_samples, all_labels, counts = [], [], []
    n_capped = 0  # re-estimation updates rejected by the step-cap (reestimate_min_cos)
    for a, b in bounds:
        win = recording.frame_slice(a, b)
        win.reset_times()
        templates = templates_from_dense(cur_dense, mask, nbefore, unit_ids, recording.channel_ids)
        mk = _window_method_kwargs(method, templates, method_kwargs=method_kwargs, wobble_factor=wobble_factor)
        mp, spikes = run_matching(win, templates, method=method, method_kwargs=mk, n_jobs=n_jobs,
                                  shape_gate_r=shape_gate_r)
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


def run_matching(recording, templates, *, method="circus-omp", method_kwargs=None, n_jobs=16,
                 shape_gate_r=None):
    """Run a matching-pursuit method; return (NumpySorting in template-unit ids, raw spikes array).

    ``shape_gate_r`` (default None = off): keep only spikes whose per-spike cosine to their ASSIGNED
    unit template is >= shape_gate_r (see per_spike_cosine). This is a SCALE-INVARIANT acceptance gate
    -- the shape analog of circus-omp's amplitude-ratio gate -- that controls precision without
    recalibration across windows / re-seeds, and (unlike an absolute objective threshold) preserves
    low-amplitude spikes. The returned spikes array is the FILTERED set, so downstream counts /
    re-estimation see only the accepted spikes.
    """
    spikes = find_spikes_from_templates(
        recording, templates, method=method, method_kwargs=method_kwargs or {},
        job_kwargs={"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": False})
    if shape_gate_r is not None:
        r = per_spike_cosine(spikes, templates, recording)
        spikes = spikes[r >= shape_gate_r]  # NaN (out-of-bounds snippet) fails the gate -> dropped
    names = spikes.dtype.names
    uf = "cluster_index" if "cluster_index" in names else "unit_index"
    samples = spikes["sample_index"].astype(np.int64)
    labels = np.asarray(templates.unit_ids)[spikes[uf].astype(np.int64)]
    mp = si.NumpySorting.from_samples_and_labels(
        [samples], [labels], sampling_frequency=FS, unit_ids=np.asarray(templates.unit_ids))
    return mp, spikes


def wobble_method_kwargs(templates, *, threshold, approx_rank=4, amplitude_variance=1.0,
                         jitter_factor=8, max_iter=1000, refractory_period_frames=10,
                         visibility_threshold=1.0, scale_min=0.0, scale_max=float("inf"),
                         engine="numpy", torch_device="cpu", shared_memory=True):
    """Build method_kwargs for run_matching(..., method="wobble").

    SpikeInterface's WobbleMatch has a DIFFERENT method_kwargs shape from circus-omp: the
    WobbleParameters fields nest under a ``parameters`` key, while engine/torch_device/shared_memory
    are siblings (find_spikes_from_templates does WobbleMatch(rec, templates=..., **method_kwargs)).

    RAW-UNIT WARNING: wobble detects on the normalized objective ``2*conv - ||t||^2`` (units of
    amplitude^2), so ``threshold`` is gain^2-scale-dependent. Our templates are RAW units
    (is_in_uV=False, gain 0.195), so the SI default threshold=50 (tuned for Neuropixels uV) is wrong
    by orders of magnitude -- hence ``threshold`` is REQUIRED (no default) and must be calibrated
    empirically (see 70_wobble_threshold_calib.py). circus-omp's ``amplitudes`` gate is, by contrast,
    scale-invariant.

    ``approx_rank`` is clamped to 4: a tetrode template has at most 4 active channels, so its spatial
    rank is <= 4; approx_rank=5 (the SI default) keeps a 5th ~zero-singular-value component (waste +
    latent nondeterminism). ``visibility_threshold`` is inert on the sparse-bank path (wobble takes
    visibility straight from the group sparsity mask), but a DENSE bank would make it live in raw
    units -- the assert below guards against that.
    """
    assert templates.are_templates_sparse(), \
        "wobble_method_kwargs expects a group-sparse bank; a dense bank changes visibility semantics"
    return {
        "parameters": {
            "threshold": float(threshold), "approx_rank": int(approx_rank),
            "amplitude_variance": float(amplitude_variance), "jitter_factor": int(jitter_factor),
            "max_iter": int(max_iter), "refractory_period_frames": int(refractory_period_frames),
            "visibility_threshold": float(visibility_threshold),
            "scale_min": float(scale_min), "scale_max": float(scale_max),
        },
        "engine": engine, "torch_device": torch_device, "shared_memory": shared_memory,
    }


def per_spike_fit(spikes, templates, recording):
    """(a, r) per matched spike against its ASSIGNED unit template, batched per tetrode.

      a = conv / ||t||^2                  -- AMPLITUDE scale (circus-omp gates exactly this, a>=0.8).
      r = conv / (||t|| * ||snippet||)    -- SCALE-INVARIANT cosine / shape match (the shape gate).

    The two come apart for low-amplitude spikes: a small-but-template-shaped spike has low ``a`` but
    high ``r`` (the cosine PRESERVES it where an amplitude/objective gate drops it), while a large
    wrong-shape fit has high ``a`` but low ``r``. Spikes whose snippet window falls outside the
    recording get a=r=NaN. Factored from 74_wobble_normgate_prototype.py so the production shape gate
    (per_spike_cosine / run_matching's shape_gate_r) and diagnostics share one implementation.
    """
    rec_groups = np.asarray(recording.get_property("group"))
    chan_ids = np.asarray(recording.channel_ids)
    ug = _unit_groups_from_mask(templates.sparsity.mask, rec_groups)  # tetrode group per template index
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)  # (n_units, n_samp, n_chan)
    nbefore, n_samp = templates.nbefore, dense.shape[1]
    nfr = recording.get_num_frames()
    uf = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
    s_all = spikes["sample_index"].astype(np.int64)
    ci_all = spikes[uf].astype(np.int64)
    g_all = ug[ci_all]
    tsq_u = np.array([float((dense[i][:, np.flatnonzero(rec_groups == ug[i])] ** 2).sum())
                      for i in range(dense.shape[0])])  # ||t||^2 per template over its active channels
    a_all = np.full(s_all.size, np.nan, dtype=np.float64)
    r_all = np.full(s_all.size, np.nan, dtype=np.float64)
    off_all = s_all - nbefore
    valid = (off_all >= 0) & (off_all + n_samp <= nfr)
    cols = np.arange(n_samp)
    for g in np.unique(g_all):
        on_g = np.flatnonzero((g_all == g) & valid)
        if on_g.size == 0:
            continue
        chans = np.flatnonzero(rec_groups == g)
        tr = np.asarray(recording.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)
        for u_idx in np.unique(ci_all[on_g]):
            sel = on_g[ci_all[on_g] == u_idx]
            snips = tr[off_all[sel][:, None] + cols[None, :], :]                 # (n_sel, n_samp, n_act)
            conv = np.einsum("ntc,tc->n", snips, dense[u_idx][:, chans])
            snip_sq = np.einsum("ntc,ntc->n", snips, snips)
            a_all[sel] = conv / tsq_u[u_idx]
            denom = np.sqrt(tsq_u[u_idx] * snip_sq)
            r_all[sel] = np.where(denom > 0, conv / denom, np.nan)
    return a_all, r_all


def per_spike_cosine(spikes, templates, recording):
    """Per-spike cosine r = cos(snippet, ASSIGNED-unit template); the scale-invariant shape gate.

    Thin wrapper over per_spike_fit (returns only r). Matcher-agnostic: run_matching's ``shape_gate_r``
    and both carry-forward loops gate on this for wobble, circus-omp, or a residual-capture pass.
    """
    return per_spike_fit(spikes, templates, recording)[1]


# ---- competitive assignment (cosine to EVERY same-tetrode template) ----------------------------
# per_spike_fit/per_spike_cosine score a spike only against its ASSIGNED template, so they can gate
# (delete) but never REASSIGN. all_template_cosines scores every same-tetrode template, returning the
# best-matching one -- the primitive both the assignment-purity scorer (_assignment_eval) and
# competitive_reassign need. Promoted from 85_tight_cosine_reassignment.py (was a one-off diagnostic).

ASYM_MS = (-0.3, 0.8)  # tight asymmetric trough window (rA): captures the trough where co-tetrode
TIGHT_SHIFT = 2        # templates differ most; +/-TIGHT_SHIFT-sample shift tolerance absorbs jitter.


def asym_window_bounds(nbefore, *, asym_ms=ASYM_MS):
    """(a, b) sample bounds of the tight asymmetric trough window around ``nbefore``."""
    a = nbefore + int(round(asym_ms[0] * FS / 1000.0))
    b = nbefore + int(round(asym_ms[1] * FS / 1000.0))
    return a, b


def all_template_cosines(win, bank, spikes, a, b, s_max, *, peak_half=15):
    """Per spike: (rF_u, rA_u to ASSIGNED unit; rF_best/arg, rA_best/arg over ALL same-tetrode templates; mad).

    rF = full-window cosine; rA = asym tight window [a:b] with +/-s_max shift-tolerance (max over shifts).
    'arg' are GLOBAL template indices (positions in ``bank.unit_ids`` == cluster_index space). Mirrors
    per_spike_fit's per-tetrode batching but scores cosines to every co-tetrode template, so the assigned
    unit (rF_u/rA_u) can be compared to the best competitor (rF_best/rA_best). Use asym_window_bounds for a,b.
    """
    rec_groups = np.asarray(win.get_property("group"))
    chan_ids = np.asarray(win.channel_ids)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    nbefore, n_samp = bank.nbefore, dense.shape[1]
    nfr = win.get_num_frames()
    noise = get_noise_levels(win, return_in_uV=False)
    uf = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
    s = spikes["sample_index"].astype(np.int64)
    ci = spikes[uf].astype(np.int64)
    g_all = ug[ci]
    off = s - nbefore
    valid = (off - s_max >= 0) & (off + n_samp + s_max <= nfr)
    ext_cols = np.arange(-s_max, n_samp + s_max)
    n = s.size
    rF_u = np.full(n, np.nan)
    rA_u = np.full(n, np.nan)
    rF_best = np.full(n, np.nan)
    rA_best = np.full(n, np.nan)
    rF_arg = np.full(n, -1, np.int64)
    rA_arg = np.full(n, -1, np.int64)
    mad = np.full(n, np.nan)
    for g in np.unique(g_all):
        on_g = np.flatnonzero((g_all == g) & valid)
        if on_g.size == 0:
            continue
        units_g = np.flatnonzero(ug == g)                  # global template indices on this tetrode
        chans = np.flatnonzero(rec_groups == g)
        nz = noise[chans]
        tr = np.asarray(win.get_traces(channel_ids=list(chan_ids[chans])), dtype=np.float32)
        ext = tr[off[on_g][:, None] + ext_cols[None, :], :]            # (m, n_samp+2*s_max, 4)
        Tf = dense[units_g][:, :, chans]                               # (nu, n_samp, 4)
        Ta = dense[units_g][:, a:b, chans]                             # (nu, W, 4)
        tsqF = np.einsum("jtc,jtc->j", Tf, Tf)
        tsqA = np.einsum("jwc,jwc->j", Ta, Ta)
        sF = ext[:, s_max:s_max + n_samp, :]
        ssqF = np.einsum("mtc,mtc->m", sF, sF)
        rF = np.einsum("mtc,jtc->mj", sF, Tf) / np.sqrt(ssqF[:, None] * tsqF[None, :])   # (m, nu)
        rA = np.full((on_g.size, units_g.size), -np.inf)
        for k in range(-s_max, s_max + 1):
            sk = ext[:, s_max + a + k:s_max + b + k, :]
            ssqk = np.einsum("mwc,mwc->m", sk, sk)
            rk = np.einsum("mwc,jwc->mj", sk, Ta) / np.sqrt(ssqk[:, None] * tsqA[None, :])
            rA = np.maximum(rA, rk)
        local_u = np.searchsorted(units_g, ci[on_g])                   # column of the assigned unit
        rows = np.arange(on_g.size)
        rF_u[on_g] = rF[rows, local_u]
        rA_u[on_g] = rA[rows, local_u]
        fb = np.argmax(rF, axis=1)
        ab = np.argmax(rA, axis=1)
        rF_best[on_g] = rF[rows, fb]
        rA_best[on_g] = rA[rows, ab]
        rF_arg[on_g] = units_g[fb]
        rA_arg[on_g] = units_g[ab]
        peak = sF[:, nbefore - peak_half:nbefore + peak_half, :]
        mad[on_g] = np.max(np.abs(peak) / nz[None, None, :], axis=(1, 2))
    return dict(rF_u=rF_u, rA_u=rA_u, rF_best=rF_best, rA_best=rA_best, rF_arg=rF_arg, rA_arg=rA_arg, mad=mad)


def competitive_reassign(win, bank, spikes, *, margin=0.0, use_tight=False, s_max=TIGHT_SHIFT,
                         cosines=None):
    """Re-label each matched spike to its best-matching SAME-TETRODE template (competitive assignment).

    The alternative to run_matching's delete-only ``shape_gate_r``: instead of DROPPING a spike whose
    cosine to its assigned template is low, OFFER it to every same-tetrode template and re-label it to the
    best match when that beats the assigned unit by >= ``margin``. ``use_tight`` scores the asym trough
    window (rA, sharper unit discrimination) rather than the full window (rF). Out-of-bounds snippets
    (NaN cosine) keep their original label. Pass precomputed ``cosines`` (all_template_cosines output for
    this spikes/bank) to skip recompute; else win+bank are used. Reuses all_template_cosines -> ZERO new
    matching compute.

    Returns (new spikes array with cluster_index/unit_index updated; sample_index unchanged, boolean mask
    of reassigned spikes). Post-hoc on any matcher output -- measure axis-B purity before vs after on the
    SAME spikes, and adjudicate each reassignment's neighbour with the CCG test (_assignment_eval) before
    folding it into the carry-forward loops.
    """
    if cosines is not None:
        cos = cosines
    else:
        a, b = asym_window_bounds(bank.nbefore)
        cos = all_template_cosines(win, bank, spikes, a, b, s_max)
    arg = cos["rA_arg" if use_tight else "rF_arg"]
    best = cos["rA_best" if use_tight else "rF_best"]
    u = cos["rA_u" if use_tight else "rF_u"]
    out = spikes.copy()
    uf = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
    ci = out[uf].astype(np.int64)
    reassign = (arg >= 0) & np.isfinite(best) & np.isfinite(u) & (arg != ci) & (best - u >= margin)
    ci[reassign] = arg[reassign]
    out[uf] = ci
    return out, reassign


def _window_method_kwargs(method, templates, *, method_kwargs, wobble_factor):
    """Per-window method_kwargs for the carry-forward loops.

    circus-omp's amplitude gate is scale-invariant, so its kwargs are STATIC across windows. wobble's
    objective is ||t||^2-scaled, so when no explicit kwargs are given its threshold is set per window
    to ``wobble_factor * tsq_median(current bank)`` (the adaptive-||t||^2 arm). For the cosine-gate arm
    pass a permissive ``wobble_factor`` (non-binding admit) plus ``shape_gate_r`` to run_matching.
    """
    if method == "wobble" and method_kwargs is None:
        if wobble_factor is None:
            raise ValueError("method='wobble' requires wobble_factor (or explicit method_kwargs)")
        return wobble_method_kwargs(templates, threshold=wobble_factor * tsq_median(templates))
    return method_kwargs


def dedup_sorting(sorting, recording, *, cosine_min=0.95, max_shift_samples=10, coincidence_ms=0.3):
    """Merge WITHIN-tetrode near-identical units (template cosine >= cosine_min) via union-find.

    Default 0.95 = the validated within-tetrode same-neuron bar: a 0.9 merge trips refractory
    contamination in ~19% of merges (conflates distinct cells), 0.95 was the safe sweet spot (see
    MATCHING_PURSUIT_FINDINGS). NOTE this is a WITHIN-WINDOW dedup (the seed window is ~stationary, so
    true oversplit twins there are HIGH-cosine); cosine is a reasonable discriminator here, unlike for
    ACROSS-TIME merges where drift drops a true duplicate's cosine (those need the temporal CCG test).

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


def windowed_carry_forward_reseed(recording, init_templates, *, window_s=1800.0, method="circus-omp",
                                  method_kwargs=None, shape_gate_r=None, wobble_factor=None,
                                  n_jobs=16, min_spikes_reestimate=100, ms_before=1.0, ms_after=2.0,
                                  reseed_every_windows=12, reseed_add_cos=0.8, reseed_min_snr=5.0,
                                  reseed_min_spikes=100, reseed_dir=None, max_windows=None):
    """PROTOTYPE: carry-forward matching with PERIODIC RE-SEEDING (the identity-swap root-cause fix).

    Like windowed_carry_forward (reestimate mode) but every ``reseed_every_windows`` windows it re-sorts
    that window with MS5 and ADDS, as NEW tracked units, any confident cluster (snr>=reseed_min_snr,
    n>=reseed_min_spikes) whose 4-channel template does NOT match an existing bank unit on its tetrode
    (shift-cos < reseed_add_cos). This gives a late-appearing / ramping neuron its OWN template so it
    claims its own spikes, instead of being captured by a same-tetrode neighbour's track (see
    MATCHING_PURSUIT_FINDINGS "IDENTITY-SWAP"; per-window step-capping cannot fix that, this can). Also
    tracks units that first appear after the seed window (fixes the seed-at-start limitation).

    Returns (assembled NumpySorting over the full recording, counts dict {unit_id: per-window array},
    births dict {unit_id: window_index at which a re-seeded unit was added}, births_cos dict
    {unit_id: max 4-ch template cosine to the existing SAME-TETRODE bank AT BIRTH}). births_cos is the
    quantity the add-cos gate thresholds on (so it is < reseed_add_cos by construction, or -1.0 when no
    same-tetrode unit existed yet): a born unit with a LOW birth_cos is a confidently distinct neuron,
    while one near the gate is a borderline duplicate of a neighbour (a drift-split twin risk) -- the
    matched-in-time signal for judging whether a finer re-seed cadence is yielding real units or twins.
    """
    import spikeinterface.sortingcomponents.matching as _m  # noqa: F401 (ensure registered)
    fs = recording.get_sampling_frequency()
    total = recording.get_num_frames()
    wlen = int(window_s * fs)
    rec_groups = np.asarray(recording.get_property("group"))
    chan_ids = recording.channel_ids
    n_chan = len(chan_ids)
    nbefore = init_templates.nbefore
    init_ids = [int(u) for u in init_templates.unit_ids]
    init_mask = init_templates.sparsity.mask
    init_dense = np.asarray(init_templates.get_dense_templates(), dtype=np.float32)
    n_samp = init_dense.shape[1]
    cur_t = {u: init_dense[i].copy() for i, u in enumerate(init_ids)}  # uid -> (n_samp, n_chan) dense
    ugroup = {u: int(rec_groups[np.flatnonzero(init_mask[i])[0]]) for i, u in enumerate(init_ids)}
    next_id = max(init_ids) + 1

    def build_bank():
        ids_now = sorted(cur_t)
        dense_now = np.stack([cur_t[u] for u in ids_now]).astype(np.float32)
        mask_now = np.stack([(rec_groups == ugroup[u]) for u in ids_now])
        return ids_now, templates_from_dense(dense_now, mask_now, nbefore, np.asarray(ids_now), chan_ids)

    bounds = [(s, min(s + wlen, total)) for s in range(0, total, wlen)]
    if max_windows:
        bounds = bounds[:max_windows]  # smoke-test knob: process only the first N windows
    nwin = len(bounds)
    all_samples, all_labels = [], []
    counts = {u: [] for u in cur_t}
    births = {}
    births_cos = {}
    for wi, (a, b) in enumerate(bounds):
        win = recording.frame_slice(a, b)
        win.reset_times()
        ids_now, templates = build_bank()
        ids_arr = np.asarray(ids_now)
        mk = _window_method_kwargs(method, templates, method_kwargs=method_kwargs, wobble_factor=wobble_factor)
        _, spikes = run_matching(win, templates, method=method, method_kwargs=mk, n_jobs=n_jobs,
                                 shape_gate_r=shape_gate_r)
        ufield = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
        s_idx = spikes["sample_index"].astype(np.int64)
        l_idx = spikes[ufield].astype(np.int64)
        labels = ids_arr[l_idx]
        all_samples.append(s_idx + a)
        all_labels.append(labels)
        cnt = np.bincount(l_idx, minlength=len(ids_now))
        for k, u in enumerate(ids_now):
            counts[u].append(int(cnt[k]))
        present = ids_arr[cnt >= min_spikes_reestimate]
        if present.size:
            ws = si.NumpySorting.from_samples_and_labels([s_idx], [labels], sampling_frequency=fs,
                                                         unit_ids=ids_arr)
            ws.set_property("group", np.array([ugroup[u] for u in ids_now]))
            ws = ws.select_units(present)
            new_t, _ = build_templates_object(ws, win, with_snr=False, n_jobs=n_jobs,
                                              ms_before=ms_before, ms_after=ms_after)
            nd = np.asarray(new_t.get_dense_templates(), dtype=np.float32)
            for k, u in enumerate([int(x) for x in new_t.unit_ids]):
                cur_t[u] = nd[k]
        if reseed_every_windows and wi > 0 and wi % reseed_every_windows == 0:
            rdir = (pathlib.Path(reseed_dir) / f"reseed_w{wi}") if reseed_dir else None
            if rdir is not None:
                shutil.rmtree(rdir, ignore_errors=True)
            rs = to_int_numpy_sorting(sort_chunk(win, rdir))
            _, raz = build_templates_object(rs, win, with_snr=True, n_jobs=n_jobs,
                                            ms_before=ms_before, ms_after=ms_after)
            rsnr = raz.get_extension("quality_metrics").get_data()["snr"].to_numpy()
            rdense = get_dense_templates_array(raz, return_in_uV=False)
            rgrp = np.asarray(rs.get_property("group"))
            n_added = 0
            for j, cu in enumerate([int(x) for x in rs.unit_ids]):
                if rsnr[j] < reseed_min_snr or len(rs.get_unit_spike_train(cu)) < reseed_min_spikes:
                    continue
                g = int(rgrp[j])
                gch = np.flatnonzero(rec_groups == g)
                ctmpl = rdense[j][:, gch]
                bank = [cur_t[u][:, gch] for u in cur_t if ugroup[u] == g]
                best_cos = max((cosine_from_templates(ctmpl, bb, max_shift_samples=10) for bb in bank),
                               default=-1.0)
                if best_cos < reseed_add_cos:  # no current same-tetrode template within add_cos -> NEW
                    full = np.zeros((n_samp, n_chan), dtype=np.float32)
                    full[:, gch] = rdense[j][:, gch]
                    cur_t[next_id] = full
                    ugroup[next_id] = g
                    counts[next_id] = [0] * (wi + 1)
                    births[next_id] = wi
                    births_cos[next_id] = float(best_cos)  # at-birth max cos to bank (gate-matched twin proxy)
                    next_id += 1
                    n_added += 1
            print(f"  [reseed w{wi} ({a / fs / 3600:.1f}h)] +{n_added} new units (bank now {len(cur_t)})",
                  flush=True)
    samples = np.concatenate(all_samples)
    labels = np.concatenate(all_labels)
    all_ids = sorted(cur_t)
    assembled = si.NumpySorting.from_samples_and_labels([samples], [labels], sampling_frequency=fs,
                                                        unit_ids=np.asarray(all_ids))
    assembled.set_property("group", np.array([ugroup[u] for u in all_ids]))
    counts_arr = {u: np.array(counts[u] + [0] * (nwin - len(counts[u]))) for u in all_ids}
    return assembled, counts_arr, births, births_cos

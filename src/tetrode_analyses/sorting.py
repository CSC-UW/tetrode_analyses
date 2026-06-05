"""MountainSort5 spike sorting for tetrode Zarr stores (sort by tetrode group).

Pipeline (lazy SpikeInterface preprocessing, per the MountainSort5 README):

    bandpass (300-6000 Hz, float32)
      -> common median reference (see `cmr`)      [full recording, before split]
      -> split_by("group")                        [one 4-ch recording per tetrode]
      -> whiten (per group, float32)
      -> MountainSort5 scheme 3 (per tetrode; filter/whiten already applied)

Common median reference (`preprocess_for_sorting(..., cmr=...)`):

- "global" (default): median across all channels. Single SI pass; ~20x faster to
  materialize than the cross-tetrode variants and benchmarked to change mean spike
  waveforms by only ~0.25% of peak (`benchmark_cmr.py`). Recommended.
- "cross_tetrode": exact median of OTHER tetrodes' channels per tetrode, via SI's
  grouped `common_reference` with per-group complement `ref_channel_ids` (the
  groupwise-cmr feature) — a single efficient pass. Requires that SI feature.
- "local": same cross-tetrode result via an annulus (geometry-based) — exact but
  slow (~64 per-channel medians). Kept for reference.

All variants preserve the ProbeGroup and group/tetrode properties needed for
split + sort.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from spikeinterface.core.baserecording import BaseRecording

# Compatibility shim: MountainSort5 0.5.8's scheme-3 block wrapper
# (mountainsort5/core/get_block_recording_for_scheme3.py) reads the old private
# attribute `recording._recording_segments`, which current SpikeInterface
# renamed to `_segments` (exposed as `.segments`). Alias it read-only so scheme 3
# works. (Everything else in ms5 already uses the public `get_num_segments()`.)
if not hasattr(BaseRecording, "_recording_segments"):
    BaseRecording._recording_segments = property(lambda self: self._segments)

# Annulus radii (um) for the cross-tetrode local median reference. exclude is
# between the max within-tetrode (20) and min between-tetrode (280) distance;
# include is larger than the max between-tetrode distance (~4520).
CMR_EXCLUDE_UM = 100.0
CMR_INCLUDE_UM = 1.0e6


def preprocess_for_sorting(
    recording,
    *,
    cmr: str = "global",
    freq_min: float = 300.0,
    freq_max: float = 6000.0,
    cmr_exclude_um: float = CMR_EXCLUDE_UM,
    cmr_include_um: float = CMR_INCLUDE_UM,
    dtype: str = "float32",
):
    """Bandpass + common median reference (full recording, lazy).

    Returns the referenced recording (still all tetrodes, `group` property intact);
    whitening is applied per group AFTER `split_by("group")`.

    ``cmr`` selects the reference:

    - ``"global"`` (default): median across ALL channels. Single SI pass, ~20x
      faster to materialize than the cross-tetrode variants; benchmarked to alter
      mean spike waveforms by ~0.25% of peak (the 4/64 self-inclusion is negligible
      for a robust median). Recommended.
    - ``"cross_tetrode"``: median of all channels on OTHER tetrodes, per tetrode
      (exact). Uses SpikeInterface's grouped ``common_reference`` with per-group
      ``ref_channel_ids`` (the groupwise-cmr feature) for a single efficient pass.
      Requires that SI feature in the active environment.
    - ``"local"``: equivalent cross-tetrode result via an annulus (geometry-based);
      exact but ~64 per-channel medians, i.e. slow. Kept for reference.
    - ``"none"``: bandpass only.
    """
    rec_f = spre.bandpass_filter(
        recording, freq_min=freq_min, freq_max=freq_max, dtype=dtype
    )
    if cmr == "none":
        return rec_f
    if cmr == "global":
        return spre.common_reference(rec_f, reference="global", operator="median", dtype=dtype)
    if cmr == "local":
        return spre.common_reference(
            rec_f, reference="local", operator="median",
            local_radius=(cmr_exclude_um, cmr_include_um), dtype=dtype,
        )
    if cmr == "cross_tetrode":
        # Per-tetrode reference = median of channels on OTHER tetrodes. Uses SI's
        # grouped global reference with per-group complement ref_channel_ids
        # (single pass, one median per group). Needs the groupwise-cmr SI feature.
        groups = np.asarray(rec_f.get_property("group"))
        cids = np.asarray(rec_f.get_channel_ids())
        uniq = sorted(set(groups.tolist()))
        group_ids = [list(cids[groups == g]) for g in uniq]
        complements = [list(cids[groups != g]) for g in uniq]
        try:
            return spre.common_reference(
                rec_f, reference="global", operator="median",
                groups=group_ids, ref_channel_ids=complements, dtype=dtype,
            )
        except (AssertionError, TypeError, IndexError) as e:
            raise NotImplementedError(
                "cmr='cross_tetrode' needs SpikeInterface's groupwise-cmr feature "
                "(global reference + per-group ref_channel_ids). Use cmr='global' or "
                "'local', or install the feature/groupwise-cmr SI build."
            ) from e
    raise ValueError(f"Unknown cmr={cmr!r}; use 'global', 'cross_tetrode', 'local', or 'none'.")


def sort_store(
    store_zarr: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    scheme: str = "3",
    scheme3_block_duration_sec: float = 1800.0,
    cmr: str = "global",
    detect_threshold: float = 5.5,
    detect_sign: int = -1,
    whitening_seed: int | None = None,
    test_duration_s: float | None = None,
    materialize_n_jobs: int = 96,
    sort_n_jobs: int = 4,
    keep_cmr_cache: bool = False,
    cmr_cache_dir: str | pathlib.Path | None = None,
    materialize_dtype: str = "float32",
    sorter_params: dict | None = None,
):
    """Sort one tetrode Zarr store with MountainSort5 scheme 3, by tetrode group.

    Returns the aggregated sorting (one `UnitsAggregationSorting` with a `group`
    property per unit).

    Cross-tetrode CMR makes each tetrode's input depend on ALL 64 channels, so
    sorting tetrodes independently from the lazy pipeline would re-read and
    re-reference the whole recording once per tetrode (~16x redundant work). To
    avoid that, the bandpass + cross-tetrode-CMR recording is **materialized
    once** to a local binary (`output_dir/_cmr_cache`, float32, all channels);
    the per-tetrode sorts then read their 4 channels cheaply from it.
    MountainSort5 applies the per-group whitening itself (`whiten=True` on each
    split 4-channel recording); we only disable its filter (already bandpassed).
    Per-group sorts run in parallel via `run_sorter_by_property`.

    ``test_duration_s`` limits to the first N seconds for a quick check.

    ``cmr_cache_dir`` overrides the default per-output ``_cmr_cache`` location with
    a shared, external one. Because bandpass + global CMR + materialize is
    deterministic, repeated sorts of the SAME store (e.g. determinism replicates)
    can point at one shared cache and skip re-materializing it; an external cache
    is owned by the caller and is never auto-deleted (``keep_cmr_cache`` is moot).

    ``materialize_dtype`` is the dtype of the materialized bandpass+CMR binary
    (default ``"float32"``). ``"int16"`` quantizes the preprocessed traces to
    integer ADC-count resolution (gain not applied at this stage), i.e. stores
    the sorter input at the original acquisition resolution -- used to test the
    effect of int16 quantization on the sort vs the float32 reference.
    """
    store_zarr = pathlib.Path(store_zarr)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rec = si.read_zarr(str(store_zarr))
    if test_duration_s is not None:
        rec = rec.frame_slice(0, int(test_duration_s * rec.get_sampling_frequency()))

    # Materialize bandpass + CMR once (resume if already cached). A shared external
    # cmr_cache_dir lets determinism replicates of the same store reuse one cache.
    external_cache = cmr_cache_dir is not None
    cmr_cache = pathlib.Path(cmr_cache_dir) if external_cache else output_dir / "_cmr_cache"
    cmr_cache.parent.mkdir(parents=True, exist_ok=True)
    pp_mat = None
    if cmr_cache.exists():
        try:
            pp_mat = si.load(str(cmr_cache))
            print(f"  resuming from cached bandpass+CMR at {cmr_cache}", flush=True)
        except Exception:
            shutil.rmtree(cmr_cache, ignore_errors=True)
    if pp_mat is None:
        pp = preprocess_for_sorting(rec, cmr=cmr)
        print(f"  materializing bandpass+CMR -> {cmr_cache} (dtype={materialize_dtype}) ...", flush=True)
        # bandpass + CMR are computed in float32; the binary is written as
        # materialize_dtype. "int16" stores the preprocessed traces at integer
        # ADC-count resolution (the original acquisition resolution; gain not
        # applied here) -- a quantization of the sorter input vs "float32".
        pp_mat = pp.save(
            format="binary", folder=str(cmr_cache), dtype=materialize_dtype,
            n_jobs=materialize_n_jobs, progress_bar=True, overwrite=True,
        )

    params = dict(
        scheme=scheme,
        filter=False,   # already bandpassed
        whiten=True,    # ms5 whitens each split 4-channel group (per-group whitening)
        whitening_seed=whitening_seed,  # int => reproducible whitening (see SI ms5 wrapper)
        scheme3_block_duration_sec=scheme3_block_duration_sec,
        detect_threshold=detect_threshold,
        detect_sign=detect_sign,
    )
    if sorter_params:
        params.update(sorter_params)

    aggregated = ss.run_sorter_by_property(
        "mountainsort5",
        pp_mat,
        grouping_property="group",
        folder=str(output_dir / "by_group"),
        engine="joblib",
        engine_kwargs={"n_jobs": sort_n_jobs},
        verbose=True,
        **params,
    )
    if not keep_cmr_cache and not external_cache:
        shutil.rmtree(cmr_cache, ignore_errors=True)
    return aggregated

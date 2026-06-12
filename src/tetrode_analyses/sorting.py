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


def materialize_preprocessed(
    store_zarr: str | pathlib.Path,
    cache_dir: str | pathlib.Path,
    *,
    cmr: str = "global",
    test_duration_s: float | None = None,
    materialize_n_jobs: int = 96,
    materialize_dtype: str = "float32",
):
    """Read a tetrode Zarr store, apply bandpass + CMR, materialize ONCE to a local
    binary cache (resume if the cache already exists).

    Shared by ``sort_store`` (MountainSort5) and ``sort_store_ks4`` (Kilosort4):
    both sorters consume the SAME bandpass(300-6000 Hz) + global-CMR float32 binary,
    so a sort of one store can reuse the other's cache byte-for-byte (an
    apples-to-apples sorter input). Returns the loaded materialized recording (all
    channels, ``group``/``tetrode`` properties + ProbeGroup intact).

    The caller owns the cache lifetime -- this function never deletes ``cache_dir``
    (a shared external cache is reused across sorts/sorters and must not be removed
    out from under a concurrent run).

    ``test_duration_s`` limits to the first N seconds (quick check / smoke test).
    ``materialize_dtype`` is the binary's dtype (``"float32"`` default; ``"int16"``
    quantizes the preprocessed traces to integer ADC-count resolution).
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        try:
            pp_mat = si.load(str(cache_dir))
            print(f"  resuming from cached bandpass+CMR at {cache_dir}", flush=True)
            return pp_mat
        except Exception:
            shutil.rmtree(cache_dir, ignore_errors=True)
    rec = si.read_zarr(str(store_zarr))
    if test_duration_s is not None:
        rec = rec.frame_slice(0, int(test_duration_s * rec.get_sampling_frequency()))
    pp = preprocess_for_sorting(rec, cmr=cmr)
    print(f"  materializing bandpass+CMR -> {cache_dir} (dtype={materialize_dtype}) ...", flush=True)
    # bandpass + CMR are computed in float32; the binary is written as
    # materialize_dtype. "int16" stores the preprocessed traces at integer ADC-count
    # resolution (the original acquisition resolution; gain not applied here) -- a
    # quantization of the sorter input vs "float32".
    return pp.save(
        format="binary", folder=str(cache_dir), dtype=materialize_dtype,
        n_jobs=materialize_n_jobs, progress_bar=True, overwrite=True,
    )


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
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize bandpass + CMR once (resume if already cached). A shared external
    # cmr_cache_dir lets determinism replicates of the same store reuse one cache.
    external_cache = cmr_cache_dir is not None
    cmr_cache = pathlib.Path(cmr_cache_dir) if external_cache else output_dir / "_cmr_cache"
    pp_mat = materialize_preprocessed(
        store_zarr, cmr_cache, cmr=cmr, test_duration_s=test_duration_s,
        materialize_n_jobs=materialize_n_jobs, materialize_dtype=materialize_dtype,
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


def sort_store_ks4(
    store_zarr: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    cmr: str = "global",
    test_duration_s: float | None = None,
    materialize_n_jobs: int = 96,
    materialize_dtype: str = "float32",
    keep_cmr_cache: bool = False,
    cmr_cache_dir: str | pathlib.Path | None = None,
    # KS4 core knobs
    batch_size: int = 300_000,
    do_CAR: bool = False,
    do_correction: bool = False,
    nblocks: int = 0,
    skip_kilosort_preprocessing: bool = False,
    torch_device: str = "auto",
    nearest_chans: int = 4,
    whitening_range: int = 4,
    dminx: float = 16.0,
    sort_n_jobs: int = 1,
    use_binary_file: bool = True,
    write_n_jobs: int = 16,
    write_chunk_duration: str = "1s",
    grouping_property: str | None = "group",
    sorter_params: dict | None = None,
):
    """Sort one tetrode Zarr store with Kilosort4, by tetrode group (the default),
    mirroring the MountainSort5 preprocessing.

    Same lazy preprocessing as ``sort_store``: bandpass(300-6000 Hz) + global CMR,
    **materialized once** (via ``materialize_preprocessed``, default float32, all 64
    channels) so the per-tetrode sorts read their 4 channels cheaply. Returns the
    aggregated sorting (one ``UnitsAggregationSorting`` with a ``group`` property per
    unit), exactly like ``sort_store``.

    KS4 differences from MS5, and why:

    - ``do_CAR=False`` -- global CMR is already applied to all 64 channels BEFORE the
      per-group split, so a second 4-channel common-average reference would
      re-reference each tetrode to itself. KS4's internal whitening
      (``skip_kilosort_preprocessing=False``) is kept as the analog of MS5's
      per-group whiten.
    - ``do_correction=False`` / ``nblocks=0`` -- NO drift correction (the SI wrapper
      sets ``ops["nblocks"]=0`` from ``do_correction=False``; ``nblocks=0`` is a
      durable belt-and-suspenders guard).
    - ``batch_size`` defaults to 300_000 samples (10 s @ 30 kHz). KS4 can fail on a
      single tetrode (4 ch) when too few spikes per batch exist to build universal
      templates; a larger batch supplies more spikes. Escalate via ``batch_size``
      (300k -> 600k -> 900k) or ``sorter_params={"templates_from_data": False}`` if
      it still fails.
    - ``nearest_chans``/``whitening_range`` default to 4 (only 4 channels exist per
      tetrode); ``dminx=16`` ~ the tetrode contact x-spread (~20 um).

    ``sort_n_jobs`` is the joblib engine parallelism across tetrodes; default 1
    (sequential) so a single KS4 process owns the GPU at a time -- clean footprint
    measurement and no CUDA contention.

    ``use_binary_file`` selects how KS4 gets each tetrode's 4 channels out of the
    sample-major (channel-interleaved) 64-channel cache, where a tetrode is a
    *strided* slice (cols 4g..4g+3 of each frame), not a contiguous sub-file:

    - ``True`` (default): KS4's fast path. The SI wrapper rewrites a contiguous
      4-channel ``recording.dat`` per tetrode (KS4's reader needs a dedicated
      N-channel file). ``write_n_jobs``/``write_chunk_duration`` parallelize that
      write (passed via the sorter params; ``split_job_kwargs`` routes them to
      ``write_binary``, KS4 ignores them) -- without it the rewrite is single-process
      and dominates wall time (~20 min/tetrode vs a few minutes at 16 jobs).
    - ``False``: no rewrite -- KS4 reads each tetrode as a read-only memmap slice of
      the 64-ch cache via ``RecordingExtractorAsArray``. Strictly less disk I/O (no
      83 GB/tetrode write + re-read), at the cost of per-batch Python read overhead.
      ``write_n_jobs`` is then unused.

    ``grouping_property`` defaults to ``"group"`` (sort each tetrode separately, like
    MS5). Set to ``None`` to sort all channels together in one KS4 run (the fallback
    when KS4 cannot get enough spikes per batch from a single tetrode).

    ``cmr_cache_dir`` / ``keep_cmr_cache`` / ``materialize_dtype`` behave exactly as
    in ``sort_store``; the cache is byte-identical to MS5's, so a KS4 run can point
    at an existing MS5 ``_cmr_cache`` (and vice versa).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    external_cache = cmr_cache_dir is not None
    cmr_cache = pathlib.Path(cmr_cache_dir) if external_cache else output_dir / "_cmr_cache"
    pp_mat = materialize_preprocessed(
        store_zarr, cmr_cache, cmr=cmr, test_duration_s=test_duration_s,
        materialize_n_jobs=materialize_n_jobs, materialize_dtype=materialize_dtype,
    )

    params = dict(
        batch_size=batch_size,
        do_CAR=do_CAR,            # global CMR already applied before the split
        do_correction=do_correction,  # SI wrapper -> ops["nblocks"]=0 when False
        nblocks=nblocks,         # explicit no-drift guard
        skip_kilosort_preprocessing=skip_kilosort_preprocessing,  # keep KS4 whitening
        nearest_chans=nearest_chans,
        whitening_range=whitening_range,
        dminx=dminx,
        torch_device=torch_device,
        use_binary_file=use_binary_file,
        # job kwargs for the per-tetrode binary rewrite (split out by
        # split_job_kwargs; KS4 itself ignores them). Without these the rewrite is
        # single-process and dominates wall time. Only used when
        # use_binary_file=True; with use_binary_file=False KS4 reads each tetrode's
        # 4 channels as a read-only memmap slice of the 64-ch cache (no rewrite).
        n_jobs=write_n_jobs,
        chunk_duration=write_chunk_duration,
    )
    if sorter_params:
        params.update(sorter_params)

    if grouping_property is not None:
        aggregated = ss.run_sorter_by_property(
            "kilosort4",
            pp_mat,
            grouping_property=grouping_property,
            folder=str(output_dir / "by_group"),
            engine="joblib",
            engine_kwargs={"n_jobs": sort_n_jobs},
            verbose=True,
            **params,
        )
    else:
        # Fallback: sort all tetrodes together (more spikes per batch for universal
        # templates). One KS4 run on all channels; returns a plain Sorting.
        aggregated = ss.run_sorter(
            "kilosort4",
            pp_mat,
            folder=str(output_dir / "all_tetrodes"),
            verbose=True,
            **params,
        )
    if not keep_cmr_cache and not external_cache:
        shutil.rmtree(cmr_cache, ignore_errors=True)
    return aggregated


def assign_tetrode_groups(sorting, recording, sorter_output_dir):
    """Attach a per-unit ``group`` (tetrode index) property to a flat KS4 sorting,
    using Kilosort4's OWN cluster->channel assignment (no template recomputation).

    Needed for the 'sort together' KS4 path (``grouping_property=None``): one
    64-channel run yields a flat sorting with no per-group structure, whereas
    ``run_sorter_by_property`` (MS5 / by-group KS4) tags each unit with its tetrode.
    KS4 stores kcoords PER CHANNEL (``channel_shanks.npy``) but not per cluster; each
    cluster's tetrode is its peak channel's tetrode. KS4's ``data_tools`` computes the
    per-cluster best channel straight from the saved ``templates.npy``
    (``(templates**2).sum(time).argmax(chan)``); SI's unit_ids are the KS4 cluster
    ids. So map best_channel -> the recording's per-channel ``group`` (== the kcoords
    KS4 was given). Returns the sorting with ``group`` set (also set in place).

    ``sorter_output_dir`` is KS4's results dir (``<folder>/sorter_output``).
    """
    from kilosort.data_tools import get_best_channels
    sorter_output_dir = pathlib.Path(sorter_output_dir)
    best_chans = np.asarray(get_best_channels(sorter_output_dir))  # per cluster_id -> channel-axis index
    chan_map = np.load(sorter_output_dir / "channel_map.npy").astype(int).ravel()  # chan axis -> recording chan index
    ch_groups = np.asarray(recording.get_property("group"))  # per recording channel -> tetrode
    unit_groups = np.array(
        [int(ch_groups[chan_map[int(best_chans[int(u)])]]) for u in sorting.unit_ids], dtype="int64"
    )
    sorting.set_property("group", unit_groups)
    return sorting

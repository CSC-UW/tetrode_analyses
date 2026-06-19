"""Track tetrode units across short overlapping sort chunks (geometry-free).

Strategy (see ``docs/plans`` / ``SORTING_COMPARISON_FINDINGS.md``): a 48 h tetrode
recording cannot be sorted in one shot (units' waveforms drift; an abrupt
discontinuity sits at ~100000 s) and cannot use geometric drift correction (the
tetrode geometry is fictional, so localization-based registration -- DARTsort /
DREDge / Kilosort drift, Yuan-EMD spatial terms -- does not apply). Instead:

1. Sort the recording in SHORT OVERLAPPING chunks with MountainSort5 scheme 2.
   Within a short chunk, waveforms are quasi-stationary -> one classifier is fine.
2. Match units across CONSECUTIVE chunks by spike-train agreement (Jaccard) in the
   OVERLAP region, where both sorts saw the *same physical spikes*. This is fully
   geometry-free and ground-truth-like. A 4-channel template cosine corroborates.
3. Chain consecutive matches transitively into global unit identities. Every link
   is a high-confidence short-range match, so a unit's template may drift
   arbitrarily over 48 h. A chain SPLITS at the ~100000 s discontinuity iff no
   overlap match bridges it -- an honest outcome if the electrode moved.

The reusable pieces here are pure library functions; the numbered scripts in
``analyses/tetrode_preprocessing_and_sorting/sorting/`` drive the experiments.

Key SpikeInterface facts this module relies on (SI 0.104.1):
- ``BaseSorting.frame_slice(start, end)`` keeps spikes in ``[start, end)`` and
  RE-ZEROS them to start at 0 -- used to crop each chunk sort to the overlap.
- ``NumpySorting.from_samples_and_labels`` / ``from_unit_dict`` take FRAMES (not
  seconds, unlike ``from_times_and_labels``).
- ``ChannelSparsity.from_property(sorting, recording, by_property="group")`` maps
  each unit to its own tetrode's 4 channels (``unit_id_to_channel_indices``).
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import time
from collections import defaultdict

import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.sorters as ss
from spikeinterface.core import ChannelSparsity

# Importing sorting also applies its MountainSort5 `_recording_segments` shim and
# gives us the shared, deterministic preprocessing.
from tetrode_analyses.sorting import preprocess_for_sorting

__all__ = [
    "Chunk",
    "plan_chunks",
    "materialize_chunk",
    "sort_chunk",
    "sort_chunk_ks4",
    "build_chunk_analyzer",
    "match_overlap",
    "group_channel_indices",
    "template_cosines",
    "extract_group_templates",
    "cosine_from_templates",
    "chain_matches",
    "heal_chains",
    "assemble_global_sorting",
    "track_span",
    "to_int_numpy_sorting",
    "shift_sorting",
]


# --------------------------------------------------------------------------- #
# Chunk planning
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Chunk:
    """One overlapping window, in absolute frames of the full recording."""

    index: int
    start_frame: int
    end_frame: int  # exclusive
    fs: float

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def t_start_s(self) -> float:
        return self.start_frame / self.fs

    @property
    def t_end_s(self) -> float:
        return self.end_frame / self.fs


def plan_chunks(
    n_frames: int,
    fs: float,
    *,
    chunk_s: float,
    overlap_frac: float,
    t0_frame: int = 0,
    t1_frame: int | None = None,
    min_tail_frac: float = 0.5,
) -> list[Chunk]:
    """Tile ``[t0_frame, t1_frame)`` into windows of ``chunk_s`` overlapping by
    ``overlap_frac`` (of the chunk length).

    ``stride = round(chunk_frames * (1 - overlap_frac))``. The final window is
    clamped to ``t1_frame``; if its length would be < ``min_tail_frac`` of a full
    chunk it is dropped and the previous window is extended to ``t1_frame`` (so
    coverage is complete and no runt chunk is sorted).
    """
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError(f"overlap_frac must be in [0, 1); got {overlap_frac}")
    t1_frame = int(n_frames if t1_frame is None else min(t1_frame, n_frames))
    chunk_frames = int(round(chunk_s * fs))
    stride = int(round(chunk_frames * (1.0 - overlap_frac)))
    if stride < 1:
        raise ValueError("overlap_frac too close to 1: stride < 1 frame")

    chunks: list[Chunk] = []
    start = int(t0_frame)
    idx = 0
    while start < t1_frame:
        end = min(start + chunk_frames, t1_frame)
        chunks.append(Chunk(idx, start, end, fs))
        if end >= t1_frame:
            break
        start += stride
        idx += 1

    # Merge a too-short tail into the previous chunk (keep full coverage).
    if len(chunks) >= 2 and chunks[-1].n_frames < int(min_tail_frac * chunk_frames):
        prev = chunks[-2]
        chunks[-2] = Chunk(prev.index, prev.start_frame, chunks[-1].end_frame, fs)
        chunks.pop()
    return chunks


# --------------------------------------------------------------------------- #
# Per-chunk materialize + sort
# --------------------------------------------------------------------------- #
def materialize_chunk(
    store_zarr: str | pathlib.Path,
    chunk: Chunk,
    out_binary_dir: str | pathlib.Path,
    *,
    cmr: str = "global",
    materialize_dtype: str = "float32",
    n_jobs: int = 96,
):
    """Materialize bandpass+CMR for ONE chunk as a small genuine-crop binary.

    Must slice the *source zarr* (not a saved binary) and ``reset_times()`` to
    avoid the ``frame_slice`` time-vector memory pitfall -- the pitfall only bites
    when ``frame_slice`` is applied to a ``BinaryFolderRecording`` whose full-length
    float64 time vector is then reloaded per worker. Slicing the zarr and saving a
    genuine crop keeps each worker's time vector chunk-sized.
    """
    rec = si.read_zarr(str(store_zarr))
    crop = rec.frame_slice(chunk.start_frame, chunk.end_frame)
    crop.reset_times()  # in-place: drop the (sliced) time vector before saving
    pp = preprocess_for_sorting(crop, cmr=cmr)
    return pp.save(
        format="binary",
        folder=str(out_binary_dir),
        dtype=materialize_dtype,
        n_jobs=n_jobs,
        progress_bar=True,
        overwrite=True,
    )


def sort_chunk(
    chunk_binary,
    out_dir: str | pathlib.Path,
    *,
    whitening_seed: int = 42,
    detect_threshold: float = 5.5,
    detect_sign: int = -1,
    scheme2_training_duration_sec: float | None = None,
    sort_n_jobs: int = 5,
):
    """Sort one already-materialized chunk binary per tetrode with MS5 scheme 2.

    Calls ``run_sorter_by_property`` directly on the chunk binary (which already
    carries the ``group`` property), mirroring the deterministic production kwargs
    in ``tetrode_analyses.sorting.sort_store``. ``scheme2_training_duration_sec``
    defaults to the whole chunk (train on everything; within a short chunk the
    waveform is ~stationary, so a single classifier is appropriate).

    Returns the aggregated chunk sorting (chunk-local frames; a ``group`` property
    per unit).
    """
    if scheme2_training_duration_sec is None:
        scheme2_training_duration_sec = chunk_binary.get_num_frames() / chunk_binary.get_sampling_frequency()
    return ss.run_sorter_by_property(
        "mountainsort5",
        chunk_binary,
        grouping_property="group",
        folder=str(pathlib.Path(out_dir) / "by_group"),
        engine="joblib",
        engine_kwargs={"n_jobs": sort_n_jobs},
        verbose=False,
        scheme="2",
        filter=False,
        whiten=True,
        whitening_seed=whitening_seed,
        scheme2_training_duration_sec=float(scheme2_training_duration_sec),
        scheme2_training_recording_sampling_mode="uniform",
        detect_threshold=detect_threshold,
        detect_sign=detect_sign,
    )


def sort_chunk_ks4(
    chunk_binary,
    out_dir: str | pathlib.Path,
    *,
    together: bool = True,
    batch_size: int = 300_000,
    templates_from_data: bool = False,
    nearest_chans: int = 4,
    whitening_range: int = 4,
    dminx: float = 16.0,
    torch_device: str = "auto",
    use_binary_file: bool = True,
    write_n_jobs: int = 16,
    sort_n_jobs: int = 1,
    sorter_params: dict | None = None,
):
    """Sort one already-materialized chunk binary with Kilosort4 (the swap-in
    counterpart to :func:`sort_chunk`'s MountainSort5 scheme 2).

    KS4 settings mirror ``tetrode_analyses.sorting.sort_store_ks4`` exactly --
    ``do_CAR=False`` (global CMR already applied), ``do_correction=False`` /
    ``nblocks=0`` (NO drift correction; the tetrode geometry is fictional, so KS4's
    drift model must stay off), KS4's internal whitening kept, ``nearest_chans =
    whitening_range = 4`` and ``dminx=16`` so detection/whitening stay within each
    tetrode. The POINT of this experiment is KS4's deconvolution / template-matching
    *within* a short, quasi-stationary chunk: unlike MS5's per-chunk clustering (which
    drops ~7% of clean units per chunk -> a 0.91 per-boundary bridge ceiling), KS4
    detects every spike that matches a template, so a unit it isolates in chunk A
    should not vanish in chunk B.

    ``templates_from_data=False`` (default; differs from KS4's own default of True)
    seeds detection with KS4's PREFAB universal templates instead of re-deriving them
    by collecting threshold-crossing clips and running ``KMeans(n_init=10)`` on them.
    The data-derived path has no clip cap and futex-hangs for hours on long, dense
    recordings (confirmed on the 48 h single-shot run). For chunked tracking it is also
    WRONG in principle: data-derived seeds differ per chunk (each chunk's KMeans gives
    different universal templates), injecting cross-chunk inconsistency, whereas the
    prefab seeds are identical across chunks. KS4 still learns the real per-unit
    templates from the data during clustering either way.

    ``together=True`` runs ONE 64-channel KS4 pass on the chunk and recovers each
    unit's tetrode by peak channel (``assign_tetrode_groups``) -- far faster than 16
    per-tetrode GPU runs (KS4 underutilizes the GPU on 4-channel data), and
    ``nearest_chans=4`` keeps units effectively per-tetrode. ``together=False`` sorts
    each tetrode separately via ``run_sorter_by_property`` (the strict MS5-parallel
    layout; slower). Returns an aggregated sorting with a per-unit ``group`` property,
    in chunk-local frames -- identical contract to :func:`sort_chunk`.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(
        batch_size=batch_size,
        templates_from_data=templates_from_data,
        do_CAR=False,
        do_correction=False,
        nblocks=0,
        skip_kilosort_preprocessing=False,
        nearest_chans=nearest_chans,
        whitening_range=whitening_range,
        dminx=dminx,
        torch_device=torch_device,
        use_binary_file=use_binary_file,
        # routed to the per-tetrode binary rewrite by split_job_kwargs; KS4 ignores them.
        n_jobs=write_n_jobs,
        chunk_duration="1s",
    )
    if sorter_params:
        params.update(sorter_params)

    if together:
        agg = ss.run_sorter(
            "kilosort4", chunk_binary, folder=str(out_dir / "all_tetrodes"), verbose=False, **params
        )
        from tetrode_analyses.sorting import assign_tetrode_groups

        return assign_tetrode_groups(agg, chunk_binary, out_dir / "all_tetrodes" / "sorter_output")
    return ss.run_sorter_by_property(
        "kilosort4",
        chunk_binary,
        grouping_property="group",
        folder=str(out_dir / "by_group"),
        engine="joblib",
        engine_kwargs={"n_jobs": sort_n_jobs},
        verbose=False,
        **params,
    )


def build_chunk_analyzer(sorting, recording, *, n_jobs: int = 5, ms_before: float = 1.0, ms_after: float = 2.0):
    """Lightweight in-memory analyzer for templates only (group sparsity).

    Computes ``random_spikes -> waveforms -> templates`` with each unit sparse to
    its own tetrode's 4 channels. Used by ``template_cosines`` for the geometry-free
    corroboration; nothing positional is computed.
    """
    sparsity = ChannelSparsity.from_property(sorting, recording, by_property="group")
    analyzer = si.create_sorting_analyzer(
        sorting, recording, format="memory", sparsity=sparsity, return_in_uV=True
    )
    analyzer.compute(
        {"random_spikes": {}, "waveforms": {"ms_before": ms_before, "ms_after": ms_after}, "templates": {}},
        n_jobs=n_jobs,
        progress_bar=False,
    )
    return analyzer


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def shift_sorting(sorting, offset_frames: int):
    """Return a ``NumpySorting`` with every spike shifted by ``offset_frames``.

    Used to put a sort made from a crop (chunk-local frames, 0-based) onto the full
    recording's absolute frame base so it can be compared against an
    absolute-framed assembled sorting. Preserves unit ids and the ``group`` property.
    """
    ud = {
        u: sorting.get_unit_spike_train(u).astype(np.int64) + int(offset_frames)
        for u in sorting.unit_ids
    }
    out = si.NumpySorting.from_unit_dict([ud], sampling_frequency=sorting.get_sampling_frequency())
    g = sorting.get_property("group")
    if g is not None:
        out.set_property("group", np.asarray(g))
    return out


def to_int_numpy_sorting(sorting):
    """Detached in-memory ``NumpySorting`` with **int64** unit ids.

    MountainSort5's aggregated sorting has uint64 unit ids (dtype kind ``"u"``),
    which ``spikeinterface.comparison`` rejects (it only accepts ``"i"``/``"U"``/
    ``"O"``). Rename positionally to ``0..n-1`` int64, preserving the ``group``
    property (kept positionally by the selection wrapper).
    """
    s = si.NumpySorting.from_sorting(sorting, with_metadata=True)
    return s.rename_units(np.arange(s.get_num_units(), dtype="int64"))


def _group_unit_ids(sorting, group: int):
    """Unit ids of ``sorting`` belonging to tetrode ``group`` (via unit property)."""
    groups = np.asarray(sorting.get_property("group"))
    uids = np.asarray(sorting.unit_ids)
    return list(uids[groups == group])


def group_channel_indices(recording, group: int) -> np.ndarray:
    """Channel INDICES (into the recording's channel array) for tetrode ``group``.

    Group sparsity gives every unit in a group the same 4 channels in this order,
    so two units' (T, 4) templates are directly comparable.
    """
    g = np.asarray(recording.get_property("group"))
    return np.where(g == group)[0]


def match_overlap(
    sorting_a,
    sorting_b,
    chunk_a: Chunk,
    chunk_b: Chunk,
    group: int,
    *,
    delta_time: float = 0.4,
    match_score: float = 0.5,
) -> list[dict]:
    """Spike-train agreement between consecutive chunks A, B within one tetrode.

    Restricts both sorts to ``group``, crops each to the overlap window (re-zeroed
    to overlap-local frames), and runs ``compare_two_sorters``. Returns one record
    per unit-A that has a best match in B::

        {"unit_a", "unit_b", "jaccard", "reciprocal"}

    ``reciprocal`` is True iff A->B and B->A are mutual best matches (Hungarian).
    Empty list if either side has no units in this group / no overlap.
    """
    ov_len = chunk_a.end_frame - chunk_b.start_frame
    if ov_len <= 0:
        return []
    a_uids = _group_unit_ids(sorting_a, group)
    b_uids = _group_unit_ids(sorting_b, group)
    if not a_uids or not b_uids:
        return []

    a_g = sorting_a.select_units(a_uids)
    b_g = sorting_b.select_units(b_uids)
    # Overlap in each chunk's LOCAL frames (sorts are chunk-local). frame_slice
    # keeps [start, end) and re-zeros, so both land in overlap-local [0, ov_len).
    a_lo = chunk_b.start_frame - chunk_a.start_frame
    a_ov = a_g.frame_slice(a_lo, a_lo + ov_len)
    b_ov = b_g.frame_slice(0, ov_len)

    cmp = sc.compare_two_sorters(
        a_ov, b_ov, sorting1_name="A", sorting2_name="B",
        delta_time=delta_time, match_score=match_score,
    )
    m_ab = cmp.get_matching()[0]  # Series: unit_a -> unit_b (-1 unmatched)
    m_ba = cmp.get_matching()[1]  # Series: unit_b -> unit_a (-1 unmatched)
    agreement = cmp.agreement_scores

    edges: list[dict] = []
    for ua, ub in m_ab.items():
        if ub == -1:
            continue
        reciprocal = bool(ub in m_ba.index and m_ba[ub] == ua)
        edges.append(
            {
                "unit_a": ua,
                "unit_b": ub,
                "jaccard": float(agreement.loc[ua, ub]),
                "reciprocal": reciprocal,
            }
        )
    return edges


def _group_template(analyzer, unit_id, group_chan_inds: np.ndarray) -> np.ndarray:
    """(T, 4) template of ``unit_id`` on its tetrode's channels.

    Handles both sparse (already 4-wide, in group-channel order) and dense (full
    width -> slice to ``group_chan_inds``) storage.
    """
    t = analyzer.get_extension("templates").get_unit_template(unit_id)
    if t.shape[1] == len(group_chan_inds):
        return t
    return t[:, group_chan_inds]


def _max_shift_cosine(ta: np.ndarray, tb: np.ndarray, max_shift: int) -> float:
    """Max cosine similarity of two (T, C) templates over integer sample shifts."""
    best = -1.0
    n = ta.shape[0]
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b = ta[s:], tb[: n - s]
        else:
            a, b = ta[: n + s], tb[-s:]
        if a.shape[0] < 1:
            continue
        af, bf = a.ravel(), b.ravel()
        na, nb = np.linalg.norm(af), np.linalg.norm(bf)
        if na == 0 or nb == 0:
            continue
        best = max(best, float(af @ bf / (na * nb)))
    return best


def extract_group_templates(analyzer, recording, groups: list[int] | None = None) -> dict:
    """Per-unit ``(T, 4)`` template on each unit's own tetrode channels.

    Geometry-free and small (~KB/unit), so these can be checkpointed to disk and the
    cross-chunk cosine recomputed later without keeping the chunk binaries. Returns
    ``{unit_id: np.ndarray (T, 4)}``.
    """
    sorting = analyzer.sorting
    if groups is None:
        groups = sorted({int(g) for g in np.asarray(recording.get_property("group"))})
    out: dict = {}
    for g in groups:
        gci = group_channel_indices(recording, g)
        for u in _group_unit_ids(sorting, g):
            out[u] = _group_template(analyzer, u, gci)
    return out


def cosine_from_templates(ta, tb, *, max_shift_samples: int = 10) -> float:
    """Max-shift cosine between two ``(T, 4)`` group templates (public wrapper)."""
    return _max_shift_cosine(np.asarray(ta), np.asarray(tb), max_shift_samples)


def cluster_is_new(cluster_template, bank_templates, *, add_cos: float = 0.8,
                   max_shift_samples: int = 10) -> bool:
    """Decide whether a re-seed cluster is a NEW unit (no match in the current bank).

    Used by periodic re-seeding (``_mp_common.windowed_carry_forward_reseed``): a re-sorted cluster is
    ADDED as a new tracked unit iff its ``(T, 4)`` template's max-shift cosine to EVERY ``bank_templates``
    entry (the existing units on the SAME tetrode, already on the same channels) is below ``add_cos``.
    Empty bank -> always new.
    """
    if len(bank_templates) == 0:
        return True
    best = max(cosine_from_templates(cluster_template, b, max_shift_samples=max_shift_samples)
               for b in bank_templates)
    return best < add_cos


def template_cosines(
    analyzer_a,
    analyzer_b,
    candidate_pairs: list[tuple],
    group_chan_inds: np.ndarray,
    *,
    max_shift_samples: int = 10,
) -> dict[tuple, float]:
    """Geometry-free 4-channel template cosine for each candidate ``(unit_a, unit_b)``.

    Both units are in the same tetrode group, so their templates share channels and
    ordering. NO positional / centroid / triangulation feature is used.
    """
    out: dict[tuple, float] = {}
    for ua, ub in candidate_pairs:
        ta = _group_template(analyzer_a, ua, group_chan_inds)
        tb = _group_template(analyzer_b, ub, group_chan_inds)
        out[(ua, ub)] = _max_shift_cosine(ta, tb, max_shift_samples)
    return out


# --------------------------------------------------------------------------- #
# Chaining + assembly
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def chain_matches(edges: list[dict], *, jaccard_min: float, cosine_min: float, nodes: list | None = None):
    """Chain accepted consecutive-chunk edges into global unit identities.

    ``edges`` items must carry: ``group, chunk_a, unit_a, chunk_b, unit_b,
    jaccard, cosine, reciprocal``. An edge is ACCEPTED iff it is reciprocal and
    ``jaccard >= jaccard_min`` and ``cosine >= cosine_min``. Union-find over
    accepted edges (consecutive pairs only -> chains, not branches). Each
    connected component is one global unit.

    ``nodes`` optionally provides the full universe of ``(group, chunk, unit)``
    tuples so that units which never matched a neighbor are retained as singleton
    globals (rather than dropped). Without it, only edge-endpoint nodes appear.

    Returns ``(node_to_global, provenance)`` where node = ``(group, chunk, unit)``,
    ``node_to_global`` maps each node to an int global id, and ``provenance`` maps
    global id -> list of member nodes sorted by chunk index.
    """
    uf = _UnionFind()
    if nodes is not None:
        for n in nodes:
            uf.add(tuple(n))
    for e in edges:
        uf.add((e["group"], e["chunk_a"], e["unit_a"]))
        uf.add((e["group"], e["chunk_b"], e["unit_b"]))
    for e in edges:
        if e["reciprocal"] and e["jaccard"] >= jaccard_min and e["cosine"] >= cosine_min:
            uf.union((e["group"], e["chunk_a"], e["unit_a"]), (e["group"], e["chunk_b"], e["unit_b"]))

    comps: dict = defaultdict(list)
    for node in uf.parent:
        comps[uf.find(node)].append(node)

    node_to_global: dict = {}
    provenance: dict = {}
    for gid, (_root, members) in enumerate(sorted(comps.items(), key=lambda kv: min(m[1] for m in kv[1]))):
        members_sorted = sorted(members, key=lambda m: (m[1], m[2]))  # by (chunk, unit)
        provenance[gid] = members_sorted
        for node in members_sorted:
            node_to_global[node] = gid
    return node_to_global, provenance


def heal_chains(
    node_to_global: dict,
    provenance: dict,
    chunk_templates: dict,
    *,
    cosine_min: float = 0.95,
    margin: float = 0.03,
    max_gap: int = 0,
    max_shift_samples: int = 10,
):
    """Global per-tetrode agglomeration of unit-segments by template similarity.

    Consecutive overlap-Jaccard matching breaks ~1 unit in 5 per boundary (MS5 chunk
    variability), and those breaks compound fatally over many boundaries — even clean
    units span only ~12 chunks. This pass reconnects the resulting segments WITHOUT
    spike overlap, using the geometry-free 4-channel template cosine (its intended
    cross-gap role). The overlap chains are the must-link anchors; here, for each
    segment that ENDS at chunk ``e``, candidates are segments that START in
    ``[e+1, e+1+max_gap]`` of the same tetrode (so a unit missed for up to ``max_gap``
    chunks is still rejoined; ``max_gap=0`` = adjacent chunks only). A merge is
    accepted only if the end↔start template cosine is **mutual-best, ≥ cosine_min, and
    an unambiguous winner** (margin ``>= margin`` over the runner-up on both sides).
    Because each accepted link has end-chunk < start-chunk, merged segments stay
    chunk-disjoint (one node per chunk) and temporally ordered, and union-find makes it
    transitive in one pass. Conservative by construction (false-merge risk on 4-channel
    tetrodes; mitigated by the overlap anchors + strict cosine).

    ``chunk_templates`` is ``{chunk_index: {unit_id: (T, 4) template}}``. Returns
    ``(new_node_to_global, new_provenance, heal_edges)`` where ``heal_edges`` is a list
    of ``(gid_end, gid_start, cosine)`` applied.
    """
    comp_start, comp_end = {}, {}
    for gid, members in provenance.items():
        ms = sorted(members, key=lambda m: (m[1], m[2]))
        comp_start[gid], comp_end[gid] = ms[0], ms[-1]
    ends_at, starts_at = defaultdict(list), defaultdict(list)
    for gid, (g, c, _u) in comp_end.items():
        ends_at[(g, c)].append(gid)
    for gid, (g, c, _u) in comp_start.items():
        starts_at[(g, c)].append(gid)

    def _tmpl(node):
        return chunk_templates[node[1]].get(node[2])

    def _best(src_node, cand_gids, cand_node_of):
        ts = _tmpl(src_node)
        if ts is None or not cand_gids:
            return None, -1.0, -1.0
        scored = []
        for gid in cand_gids:
            tt = _tmpl(cand_node_of[gid])
            if tt is not None:
                scored.append((cosine_from_templates(ts, tt, max_shift_samples=max_shift_samples), gid))
        if not scored:
            return None, -1.0, -1.0
        scored.sort(reverse=True)
        best_cos, best_gid = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        return best_gid, best_cos, second

    heal_edges = []
    for gid_a, (g, e, _u) in comp_end.items():
        cand_starts = [gid for k in range(1, max_gap + 2) for gid in starts_at.get((g, e + k), [])]
        b_gid, b_cos, b_2nd = _best(comp_end[gid_a], cand_starts, comp_start)
        if b_gid is None or b_cos < cosine_min or not (b_2nd < 0 or b_cos - b_2nd >= margin):
            continue
        # reciprocity: does b_gid's start pick gid_a as its unambiguous best end in window?
        s = comp_start[b_gid][1]
        cand_ends = [gid for k in range(1, max_gap + 2) for gid in ends_at.get((g, s - k), [])]
        a_gid, a_cos, a_2nd = _best(comp_start[b_gid], cand_ends, comp_end)
        if a_gid != gid_a or not (a_2nd < 0 or a_cos - a_2nd >= margin):
            continue
        heal_edges.append((gid_a, b_gid, b_cos))

    uf = _UnionFind()
    for gid in provenance:
        uf.add(gid)
    for a, b, _c in heal_edges:
        uf.union(a, b)
    roots = sorted({uf.find(gid) for gid in provenance})
    root_to_new = {r: i for i, r in enumerate(roots)}
    new_node_to_global = {node: root_to_new[uf.find(old)] for node, old in node_to_global.items()}
    new_prov: dict = defaultdict(list)
    for node, gid in new_node_to_global.items():
        new_prov[gid].append(node)
    for gid in new_prov:
        new_prov[gid] = sorted(new_prov[gid], key=lambda m: (m[1], m[2]))
    return new_node_to_global, dict(new_prov), heal_edges


def _ownership_bounds(chunks: list[Chunk]) -> dict[int, tuple[int, int]]:
    """Assign each absolute frame to exactly one chunk, split at overlap midpoints.

    Chunk i owns ``[lo_i, hi_i)``: ``lo_i`` is the midpoint of its overlap with
    i-1 (or its own start for the first), ``hi_i`` the midpoint of its overlap with
    i+1 (or its own end for the last). De-duplicates coincident spikes detected in
    the overlap by both chunks.
    """
    bounds: dict[int, tuple[int, int]] = {}
    for i, ch in enumerate(chunks):
        lo = ch.start_frame
        hi = ch.end_frame
        if i > 0:
            prev = chunks[i - 1]
            lo = (ch.start_frame + prev.end_frame) // 2  # midpoint of overlap with prev
        if i < len(chunks) - 1:
            nxt = chunks[i + 1]
            hi = (nxt.start_frame + ch.end_frame) // 2  # midpoint of overlap with next
        bounds[ch.index] = (lo, hi)
    return bounds


def assemble_global_sorting(
    chunk_sortings: dict[int, "si.BaseSorting"],
    chunks: list[Chunk],
    node_to_global: dict,
    *,
    fs: float,
):
    """Splice chained chunk units into one global ``NumpySorting`` (absolute frames).

    For each member node ``(group, chunk, unit)`` of a global unit, take that unit's
    chunk-local spike train, restrict to the chunk's *owned* frame range (overlap
    de-dup via ``_ownership_bounds``), and shift to absolute frames. Concatenate
    across the global unit's contiguous member chunks.

    Returns ``(global_sorting, unit_groups)`` where ``global_sorting`` is a
    single-segment ``NumpySorting`` with integer global unit ids and a ``group``
    property, and ``unit_groups`` maps global id -> tetrode group.
    """
    chunk_by_index = {ch.index: ch for ch in chunks}
    bounds = _ownership_bounds(chunks)

    per_global_frames: dict[int, list[np.ndarray]] = defaultdict(list)
    unit_groups: dict[int, int] = {}
    for node, gid in node_to_global.items():
        group, cidx, uid = node
        ch = chunk_by_index[cidx]
        lo, hi = bounds[cidx]
        st = chunk_sortings[cidx].get_unit_spike_train(uid)  # chunk-local frames
        abs_st = st.astype(np.int64) + ch.start_frame
        owned = abs_st[(abs_st >= lo) & (abs_st < hi)]
        per_global_frames[gid].append(owned)
        unit_groups[gid] = group

    units_dict: dict[int, np.ndarray] = {}
    for gid, parts in per_global_frames.items():
        frames = np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        units_dict[gid] = frames

    global_sorting = si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)
    groups_in_order = [unit_groups[u] for u in global_sorting.unit_ids]
    global_sorting.set_property("group", np.asarray(groups_in_order))
    return global_sorting, unit_groups


# --------------------------------------------------------------------------- #
# End-to-end orchestrator
# --------------------------------------------------------------------------- #
def track_span(
    store_zarr: str | pathlib.Path,
    work_dir: str | pathlib.Path,
    *,
    chunk_s: float,
    overlap_frac: float,
    t0_frame: int = 0,
    t1_frame: int | None = None,
    groups: list[int] | None = None,
    jaccard_min: float = 0.5,
    cosine_min: float = 0.9,
    match_score: float = 0.5,
    delta_time: float = 0.4,
    cmr: str = "global",
    materialize_n_jobs: int = 96,
    sort_n_jobs: int = 5,
    sorter: str = "ms5",
    ks4_kwargs: dict | None = None,
    max_shift_samples: int = 10,
    keep_binaries: bool = False,
    log=print,
) -> dict:
    """Run the full chunk -> sort -> match -> chain -> assemble pipeline over a span.

    Returns a dict with: ``chunks``, ``chunk_sortings`` (in-memory ``NumpySorting``
    per chunk index), ``edges`` (all consecutive-chunk candidate edges with jaccard
    + cosine + reciprocal), ``node_to_global``, ``provenance``, ``global_sorting``,
    ``unit_groups``, ``fs``.

    Memory/disk: only two chunk analyzers are alive at once; each chunk binary is
    deleted once its analyzer is built (templates are held in memory), and each
    chunk sorting is detached to an in-memory ``NumpySorting`` so the sorter output
    folder can be removed too (unless ``keep_binaries``). The matching thresholds
    (``jaccard_min``, ``cosine_min``) only affect chaining, so a caller can re-run
    :func:`chain_matches` over the returned ``edges`` to sweep thresholds for free.
    """
    store_zarr = pathlib.Path(store_zarr)
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rec = si.read_zarr(str(store_zarr))
    fs = rec.get_sampling_frequency()
    nfr = rec.get_num_frames()
    chunks = plan_chunks(nfr, fs, chunk_s=chunk_s, overlap_frac=overlap_frac, t0_frame=t0_frame, t1_frame=t1_frame)
    if groups is None:
        groups = sorted({int(g) for g in np.asarray(rec.get_property("group"))})
    gci = {g: group_channel_indices(rec, g) for g in groups}

    log(f"[track_span] {len(chunks)} chunks x {len(groups)} groups | chunk_s={chunk_s} overlap={overlap_frac}")

    chunk_sortings: dict[int, "si.BaseSorting"] = {}
    edges: list[dict] = []
    prev = None  # (chunk, sorting, analyzer)
    for ch in chunks:
        t0 = time.perf_counter()
        bin_dir = work_dir / f"chunk{ch.index:03d}_bin"
        sort_dir = work_dir / f"chunk{ch.index:03d}_sort"
        cb = materialize_chunk(store_zarr, ch, bin_dir, cmr=cmr, n_jobs=materialize_n_jobs)
        if sorter == "ks4":
            srt = sort_chunk_ks4(cb, sort_dir, **(ks4_kwargs or {}))
        else:
            srt = sort_chunk(cb, sort_dir, sort_n_jobs=sort_n_jobs)
        srt_mem = to_int_numpy_sorting(srt)  # detach from disk + int64 ids for comparison
        analyzer = build_chunk_analyzer(srt_mem, cb, n_jobs=sort_n_jobs)  # keyed by the same int ids
        chunk_sortings[ch.index] = srt_mem

        if prev is not None:
            pch, psrt, pan = prev
            for g in groups:
                e_spikes = match_overlap(psrt, srt_mem, pch, ch, g, delta_time=delta_time, match_score=match_score)
                if not e_spikes:
                    continue
                pairs = [(e["unit_a"], e["unit_b"]) for e in e_spikes]
                cos = template_cosines(pan, analyzer, pairs, gci[g], max_shift_samples=max_shift_samples)
                for e in e_spikes:
                    edges.append(
                        {
                            "group": g,
                            "chunk_a": pch.index,
                            "unit_a": e["unit_a"],
                            "chunk_b": ch.index,
                            "unit_b": e["unit_b"],
                            "jaccard": e["jaccard"],
                            "cosine": cos[(e["unit_a"], e["unit_b"])],
                            "reciprocal": e["reciprocal"],
                        }
                    )
        prev = (ch, srt_mem, analyzer)
        if not keep_binaries:
            shutil.rmtree(bin_dir, ignore_errors=True)
            shutil.rmtree(sort_dir, ignore_errors=True)
        log(
            f"[track_span] chunk {ch.index} [{ch.t_start_s:.0f}-{ch.t_end_s:.0f}s] "
            f"{srt_mem.get_num_units()} units in {(time.perf_counter() - t0) / 60:.1f} min"
        )

    node_to_global, provenance = chain_matches(edges, jaccard_min=jaccard_min, cosine_min=cosine_min)
    global_sorting, unit_groups = assemble_global_sorting(chunk_sortings, chunks, node_to_global, fs=fs)
    log(f"[track_span] -> {global_sorting.get_num_units()} global units from {len(edges)} candidate edges")
    return dict(
        chunks=chunks,
        chunk_sortings=chunk_sortings,
        edges=edges,
        node_to_global=node_to_global,
        provenance=provenance,
        global_sorting=global_sorting,
        unit_groups=unit_groups,
        fs=fs,
    )

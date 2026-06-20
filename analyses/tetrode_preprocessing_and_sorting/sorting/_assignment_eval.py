"""Axis-B (per-unit ASSIGNMENT PURITY) scoring for tetrode matching-pursuit sortings.

This is the metric the wobble / coverage work drifted away from: are a unit's spikes assigned to the
CORRECT same-tetrode unit, not merely to *some* unit on the right tetrode? Both proxies the later work
optimized are BLIND to it -- an independently-firing same-tetrode neighbour mis-assigned into a unit
raises neither:
  * tetrode-pooled coverage (``_wobble_eval.coverage_by_band`` -> ``_by_tet``) counts the spike as
    "claimed" because it landed on the right tetrode;
  * median ``rp_contamination`` (``_wobble_eval.precision_summary``) sees no refractory violation,
    because a *distinct* neighbour fires independently of the host unit.
So axis B needs its own machinery.

NOTHING here is ground truth -- no curated labels exist on this 48 h tetrode recording. Instead we
TRIANGULATE three INDEPENDENT internal signals, each with a different failure mode, and flag a unit only
when >= 2 agree (the >= 2-of-3 rule + thresholds are PROVISIONAL tuning knobs, to be calibrated against
the known crowded-tetrode units u66/u79 from script 85; see the plan):

  1. cosine self-consistency -- ``per_unit_best_match_purity``: fraction of a unit's spikes whose
     ASSIGNED template is the best same-tetrode cosine match (built on ``_mp_common.all_template_cosines``).
     FAILS: amplitude-blind, cannot tell a harmless oversplit twin from a distinct contaminating
     neighbour; biased toward the sharper template.
  2. CCG cross-contamination -- ``ccg_cross_contamination``: refractory-DIP-vs-FILLED central/flank ratio
     between same-tetrode units (the temporal test cosine cannot do; promoted from
     ``63_reseed_twin_adjudication.py``). FAILS: abstains (SEGREGATED) on low co-activity and on clean
     temporal hand-offs; low-rate units never accumulate enough flank coincidences.
  3. held-out-window agreement -- ``heldout_window_agreement``: build each template from EVEN windows,
     ask whether its ODD-window spikes still match it best. Orthogonal (disjoint data). FAILS: strong
     drift across the split lowers a clean drifting unit's self-frac (confounds with contamination).

``assignment_purity_summary`` combines them per unit and applies the flag, annotating each flagged unit
as cross-contaminated (distinct neighbour, real) vs oversplit (duplicate neighbour, an axis-C merge
issue) so disagreements stay human-adjudicable.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import spikeinterface as si

from _mp_common import (FS, TIGHT_SHIFT, _unit_groups_from_mask, all_template_cosines,
                        asym_window_bounds, build_templates_object)

# CCG verdict thresholds -- promoted VERBATIM from 63_reseed_twin_adjudication.py; the defaults below are
# bit-identical to that script, so the arbiter's behaviour is unchanged (regression-tested for parity).
REFR_MS = 1.5           # refractory half-window for the central CCG bin
FLANK_MS = (5.0, 25.0)  # flank lag band
MAXLAG_MS = 25.0
DIP, FILL = 0.30, 0.70  # ratio verdict thresholds (dip -> duplicate, filled -> distinct)
MIN_CO_WIN, MIN_FLANK = 2, 30  # need this much co-activity for refractory to decide


# ---- signal 1: cosine self-consistency --------------------------------------------------------

def per_unit_best_match_purity(spikes, cosines, bank_unit_ids, *, use_tight=False):
    """Per unit: fraction of its spikes whose ASSIGNED template is the best same-tetrode cosine match.

    ``cosines`` = ``all_template_cosines`` output. A spike is "pure" when its best same-tetrode template
    (full-window ``rF_arg``, or tight-window ``rA_arg`` if ``use_tight``) IS the assigned unit; a
    "neighbour-win" is a spike that matches a DIFFERENT same-tetrode template better -- the cross-unit
    mis-assignment ``rp_contamination`` is blind to. Out-of-bounds spikes (arg == -1) are excluded.

    Returns ``{unit_id: {n, n_finite, best_match_frac, neighbor_win_frac, top_neighbor,
    top_neighbor_frac}}``. ``top_neighbor`` is the unit_id that most often out-matches this unit (-1 if
    none); ``top_neighbor_frac`` is that neighbour's share of the unit's finite spikes.
    """
    uf = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
    ci = spikes[uf].astype(np.int64)
    arg = cosines["rA_arg" if use_tight else "rF_arg"]
    out = {}
    for u in np.unique(ci):
        uid = int(bank_unit_ids[u])
        n_all = int((ci == u).sum())
        sel = (ci == u) & (arg >= 0)
        n_fin = int(sel.sum())
        if n_fin == 0:
            out[uid] = dict(n=n_all, n_finite=0, best_match_frac=float("nan"),
                            neighbor_win_frac=float("nan"), top_neighbor=-1, top_neighbor_frac=float("nan"))
            continue
        a_sel = arg[sel]
        is_best = a_sel == u
        nb = a_sel[~is_best]
        if nb.size:
            vals, cnts = np.unique(nb, return_counts=True)
            top = int(bank_unit_ids[vals[np.argmax(cnts)]])
            top_frac = float(cnts.max() / n_fin)
        else:
            top, top_frac = -1, 0.0
        out[uid] = dict(n=n_all, n_finite=n_fin, best_match_frac=float(is_best.mean()),
                        neighbor_win_frac=float((~is_best).mean()), top_neighbor=top,
                        top_neighbor_frac=top_frac)
    return out


# ---- signal 1, windowed: build per-window cosines + accumulate across windows -----------------
# all_template_cosines reads the WHOLE recording's tetrode traces, so it is window-scale (script 85 ran
# it on 1800 s windows). To score a 48 h assembled sorting on axis B without holding 48 h of trace in
# RAM, walk windows: window_assignment_cosines builds window-LOCAL templates (drift-appropriate -- the
# way the carry-forward produced the assignments) and scores that window's spikes; accumulate_best_match
# folds each window into a cross-window counter; finalize_best_match emits the same per-unit dict shape as
# per_unit_best_match_purity.

def window_bank(recording, sorting, a_frame, b_frame, *, n_jobs=16, min_spikes_template=50, seed=0):
    """Window [a_frame, b_frame): the sliced+reset recording, a window-local Templates bank built from the
    sorting's window-local spikes (drift-appropriate), and {unit_id: window-local frame train} for units
    with >= min_spikes_template spikes here. Returns (win, bank, trains); bank is None (trains={}) if no
    unit qualifies. Shared by window_assignment_cosines (axis-B purity) and residual capture (script 87).
    """
    win = recording.frame_slice(a_frame, b_frame)
    win.reset_times()
    groups = np.asarray(sorting.get_property("group"))
    gmap = {int(u): int(groups[i]) for i, u in enumerate(sorting.unit_ids)}
    trains = {}
    for u in sorting.unit_ids:
        tr = sorting.get_unit_spike_train(u).astype(np.int64)
        tr = tr[(tr >= a_frame) & (tr < b_frame)] - a_frame
        if tr.size >= min_spikes_template:
            trains[int(u)] = np.sort(tr)
    if not trains:
        return win, None, {}
    keep = sorted(trains)
    sub = si.NumpySorting.from_unit_dict([{u: trains[u] for u in keep}], sampling_frequency=FS)
    sub.set_property("group", np.array([gmap[u] for u in keep]))
    bank, _ = build_templates_object(sub, win, with_snr=False, n_jobs=n_jobs, seed=seed)
    return win, bank, trains


def window_assignment_cosines(recording, sorting, a_frame, b_frame, *, n_jobs=16, s_max=TIGHT_SHIFT,
                              min_spikes_template=50, seed=0):
    """For window [a_frame, b_frame): build window-local templates (``window_bank``) and score each window
    spike's cosine to every same-tetrode template. Returns (spikes_local, cosines, bank_unit_ids), or
    (None, None, None) if no unit has >= min_spikes_template here.
    """
    win, bank, trains = window_bank(recording, sorting, a_frame, b_frame, n_jobs=n_jobs,
                                    min_spikes_template=min_spikes_template, seed=seed)
    if bank is None:
        return None, None, None
    bank_ids = np.asarray([int(u) for u in bank.unit_ids])
    samp, ci = [], []
    for i, u in enumerate(bank_ids):
        t = trains[int(u)]
        samp.append(t)
        ci.append(np.full(t.size, i, np.int64))
    spikes = np.zeros(int(sum(len(s) for s in samp)),
                      dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    spikes["sample_index"] = np.concatenate(samp)
    spikes["cluster_index"] = np.concatenate(ci)
    a, b = asym_window_bounds(bank.nbefore)
    cosines = all_template_cosines(win, bank, spikes, a, b, s_max)
    return spikes, cosines, bank_ids


def accumulate_best_match(acc, spikes, cosines, bank_unit_ids, *, use_tight=False):
    """Fold one window's per-spike best-match cosines into a cross-window accumulator (mutates + returns
    ``acc``). ``acc`` maps unit_id -> [n, n_finite, n_best, Counter(neighbor_id -> count)]; uses GLOBAL
    unit ids so per-window banks (which drop units below the template floor) stay consistent."""
    uf = "cluster_index" if "cluster_index" in spikes.dtype.names else "unit_index"
    ci = spikes[uf].astype(np.int64)
    arg = cosines["rA_arg" if use_tight else "rF_arg"]
    for u in np.unique(ci):
        uid = int(bank_unit_ids[u])
        rec = acc.setdefault(uid, [0, 0, 0, Counter()])
        sel = ci == u
        rec[0] += int(sel.sum())
        a_sel = arg[sel & (arg >= 0)]
        rec[1] += int(a_sel.size)
        is_best = a_sel == u
        rec[2] += int(is_best.sum())
        for nb in a_sel[~is_best]:
            rec[3][int(bank_unit_ids[nb])] += 1
    return acc


def finalize_best_match(acc):
    """Turn an ``accumulate_best_match`` accumulator into the per-unit dict shape of
    ``per_unit_best_match_purity`` (best_match_frac, neighbor_win_frac, top_neighbor, ...)."""
    out = {}
    for uid, (n, n_fin, n_best, nbrs) in acc.items():
        if n_fin == 0:
            out[uid] = dict(n=n, n_finite=0, best_match_frac=float("nan"),
                            neighbor_win_frac=float("nan"), top_neighbor=-1, top_neighbor_frac=float("nan"))
            continue
        top, top_c = (nbrs.most_common(1)[0] if nbrs else (-1, 0))
        out[uid] = dict(n=n, n_finite=n_fin, best_match_frac=float(n_best / n_fin),
                        neighbor_win_frac=float(1 - n_best / n_fin), top_neighbor=int(top),
                        top_neighbor_frac=float(top_c / n_fin))
    return out


# ---- propose: best same-tetrode template for arbitrary (unassigned) events --------------------

def best_template_for_events(win, bank, event_samples, event_groups, *, s_max=TIGHT_SHIFT):
    """Best-matching same-tetrode template (full-window cosine) for arbitrary detected events that carry
    NO assignment -- the 'cosine proposes' step of residual capture. Reuses all_template_cosines by giving
    each event a dummy assigned cluster on its OWN tetrode (so the same-tetrode candidate set is correct;
    the cosine-to-assigned rF_u is ignored). Returns (best_cos, best_unit_id, valid): best_unit_id is a
    bank unit_id, -1 where the snippet is out of bounds or its tetrode has no template.
    """
    rec_groups = np.asarray(win.get_property("group"))
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    bank_ids = np.asarray(bank.unit_ids)
    rep: dict = {}
    for i, g in enumerate(ug):
        rep.setdefault(int(g), i)  # a representative bank index per tetrode group
    ev_s = np.asarray(event_samples, np.int64)
    ev_g = np.asarray(event_groups, np.int64)
    ci = np.array([rep.get(int(g), 0) for g in ev_g], np.int64)
    spikes = np.zeros(ev_s.size, dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    spikes["sample_index"] = ev_s
    spikes["cluster_index"] = ci
    a, b = asym_window_bounds(bank.nbefore)
    cos = all_template_cosines(win, bank, spikes, a, b, s_max)
    arg = cos["rF_arg"]
    valid = np.array([int(g) in rep for g in ev_g]) & (arg >= 0)
    best_uid = np.where(valid, bank_ids[arg.clip(0)], -1)
    return cos["rF_best"], best_uid, valid


# ---- signal 2: CCG cross-contamination (promoted from script 63) ------------------------------

def ccg_lags(tb, tu, maxlag):
    """All (tu - tb) lags within +/-maxlag frames, for sorted frame arrays tb, tu."""
    if len(tb) == 0 or len(tu) == 0:
        return np.empty(0)
    lo = np.searchsorted(tu, tb - maxlag)
    hi = np.searchsorted(tu, tb + maxlag)
    return np.concatenate([tu[a:b] - s for s, a, b in zip(tb, lo, hi) if b > a]) if np.any(hi > lo) \
        else np.empty(0)


def adjudicate(tb, tu, *, fs=FS, refr_ms=REFR_MS, flank_ms=FLANK_MS, maxlag_ms=MAXLAG_MS,
               min_flank=MIN_FLANK):
    """Cross-correlogram central/flank ratio (+central, flank counts) for two co-restricted frame trains.

    ratio = (coincidences within +/-refr_ms, per ms) / (coincidences in the flank band, per ms). NaN if
    fewer than ``min_flank`` flank coincidences (refractory can't decide). Promoted from script 63.
    """
    maxlag = int(maxlag_ms * 1e-3 * fs)
    lags = ccg_lags(np.sort(tb), np.sort(tu), maxlag)
    al = np.abs(lags) / fs * 1000.0
    central = int(np.sum(al <= refr_ms))
    flank = int(np.sum((al >= flank_ms[0]) & (al <= flank_ms[1])))
    cw, fw = 2 * refr_ms, 2 * (flank_ms[1] - flank_ms[0])
    if flank < min_flank:
        return np.nan, central, flank
    return (central / cw) / (flank / fw), central, flank


def verdict_of(ratio, n_co, *, dip=DIP, fill=FILL, min_co_win=MIN_CO_WIN):
    """Map a CCG ratio + co-active-window count to DUPLICATE / DISTINCT / AMBIGUOUS / SEGREGATED."""
    if n_co < min_co_win or not np.isfinite(ratio):
        return "SEGREGATED"
    if ratio < dip:
        return "duplicate"
    if ratio > fill:
        return "distinct"
    return "ambiguous"


def co_restrict(train, co_wins, wlen):
    """Restrict a frame train to the window indices in ``co_wins`` (where both units are co-active)."""
    return train[np.isin(train // wlen, co_wins)] if len(train) else train


def ccg_verdict_pair(train_a, train_b, *, win_frames, min_co_spikes=5, **kw):
    """Co-restrict two trains to their shared (>= min_co_spikes each) windows, then CCG ratio + verdict.

    The shared "cosine proposes, CCG disposes" primitive (RESIDUAL_CAPTURE_PLAN s11): given a candidate
    same-tetrode pair (e.g. a neighbour-win from ``per_unit_best_match_purity``), decide DUPLICATE (same
    cell, refractory dip -> harmless oversplit / merge candidate) vs DISTINCT (independent, filled CCG ->
    real cross-unit contamination) vs AMBIGUOUS / SEGREGATED (refractory cannot decide).
    """
    ta = np.sort(np.asarray(train_a, np.int64))
    tb = np.sort(np.asarray(train_b, np.int64))
    nwin = int(max(ta[-1] if ta.size else 0, tb[-1] if tb.size else 0) // win_frames) + 1
    na = np.bincount(ta // win_frames, minlength=nwin)
    nb = np.bincount(tb // win_frames, minlength=nwin)
    co = np.flatnonzero((na >= min_co_spikes) & (nb >= min_co_spikes))
    ca = co_restrict(ta, co, win_frames)
    cb = co_restrict(tb, co, win_frames)
    ratio, central, flank = adjudicate(cb, ca, **kw)
    return dict(ratio=float(ratio), n_co=int(co.size), central=int(central), flank=int(flank),
                verdict=verdict_of(ratio, co.size))


def ccg_cross_contamination(sorting, pairs=None, *, win_s=1800.0, fs=FS, **kw):
    """CCG verdict for same-tetrode unit pairs. ``pairs`` = list of (uid_a, uid_b); if None, ALL
    same-tetrode pairs. Returns ``{(uid_a, uid_b): verdict-dict}``. A 'distinct' verdict on a high-cosine
    neighbour-win (from signal 1) is real cross-unit contamination; 'duplicate' is oversplit (axis C).
    """
    win_frames = int(win_s * fs)
    trains = {int(u): np.sort(sorting.get_unit_spike_train(u)).astype(np.int64) for u in sorting.unit_ids}
    if pairs is None:
        groups = np.asarray(sorting.get_property("group"))
        gmap = {int(u): int(groups[i]) for i, u in enumerate(sorting.unit_ids)}
        by_g: dict = {}
        for u, g in gmap.items():
            by_g.setdefault(g, []).append(u)
        pairs = [(a, b) for members in by_g.values()
                 for i, a in enumerate(members) for b in members[i + 1:]]
    out = {}
    for a, b in pairs:
        if int(a) not in trains or int(b) not in trains:
            continue
        out[(int(a), int(b))] = ccg_verdict_pair(trains[int(a)], trains[int(b)], win_frames=win_frames, **kw)
    return out


# ---- merge-first: CCG-guarded within-tetrode merge of oversplit duplicate tracks --------------

def ccg_guarded_merge(sorting, recording, *, propose_cos=0.90, fallback_cos=0.95, win_s=1800.0,
                      max_shift_samples=10, n_jobs=16, coincidence_ms=0.3, fs=FS):
    """Merge within-tetrode OVERSPLIT duplicate tracks -- the 'merge-first' step (the dominant defect per
    the 2026-06-19 re-score: ~110 CCG-duplicate same-tetrode pairs in the 109-unit base).

    Cosine PROPOSES candidate same-tetrode pairs (template cosine >= ``propose_cos``); the CCG arbiter
    DISPOSES: accept a merge iff the pair's refractory-dip CCG verdict is 'duplicate' (same cell) OR the
    cosine >= ``fallback_cos`` (the validated strict bar, used where CCG ABSTAINS -- 'SEGREGATED' -- on too
    little co-activity). Safer than cosine-only dedup (``_mp_common.dedup_sorting``): cosine is
    amplitude-blind, and a 0.90-only merge conflates distinct co-located cells (~19% refractory-contaminated
    per the dedup findings); the CCG guard rejects those while the cosine fallback still catches genuine
    duplicates CCG cannot evaluate. Returns (merged NumpySorting [int ids, 'group' preserved, coincident
    spikes < coincidence_ms collapsed], list of accepted-merge records (a, b, verdict, cosine)).
    """
    from tetrode_analyses.tracking import _UnionFind, cosine_from_templates
    templates, _ = build_templates_object(sorting, recording, with_snr=False, n_jobs=n_jobs)
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)
    mask = templates.sparsity.mask
    uids = [int(u) for u in sorting.unit_ids]
    groups = np.asarray(sorting.get_property("group"))
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    trains = {u: np.sort(sorting.get_unit_spike_train(u).astype(np.int64)) for u in uids}
    win_frames = int(win_s * fs)
    by_group: dict = {}
    for u, g in zip(uids, groups):
        by_group.setdefault(int(g), []).append(u)
    union = _UnionFind()
    for u in uids:
        union.add(u)
    merges = []
    for members in by_group.values():
        for ia in range(len(members)):
            a = members[ia]
            ch = np.flatnonzero(mask[uid_to_idx[a]])
            ta = dense[uid_to_idx[a]][:, ch]
            for ib in range(ia + 1, len(members)):
                b = members[ib]
                tb = dense[uid_to_idx[b]][:, ch]
                cos = cosine_from_templates(ta, tb, max_shift_samples=max_shift_samples)
                if cos < propose_cos:
                    continue
                verdict = ccg_verdict_pair(trains[a], trains[b], win_frames=win_frames)["verdict"]
                if verdict == "duplicate" or cos >= fallback_cos:
                    union.union(a, b)
                    merges.append((a, b, verdict, float(cos)))
    comp: dict = {}
    for u in uids:
        comp.setdefault(union.find(u), []).append(u)
    tol = int(coincidence_ms * 1e-3 * fs)
    new_trains, new_groups = {}, []
    for new_id, mem in enumerate(comp.values()):
        allspk = np.sort(np.concatenate([trains[u] for u in mem]))
        if allspk.size and tol > 0:
            allspk = allspk[np.concatenate([[True], np.diff(allspk) > tol])]
        new_trains[new_id] = allspk
        new_groups.append(int(groups[uid_to_idx[mem[0]]]))
    merged = si.NumpySorting.from_unit_dict([new_trains], sampling_frequency=fs)
    merged.set_property("group", np.asarray(new_groups))
    return merged, merges


# ---- signal 3: held-out-window agreement (internal, no matcher re-run) -------------------------

def heldout_window_agreement(recording, sorting, *, win_s=1800.0, fs=FS, n_jobs=16,
                             min_spikes_each_half=25, max_test_spikes=4000, s_max=2, seed=0):
    """Build each unit's template from its EVEN windows; check its ODD-window spikes still match it best.

    Orthogonal to signals 1/2 (templates come from DISJOINT data; the test is purely shape-on-held-out).
    Reuses ``build_templates_object`` (even-window sub-sorting) + ``all_template_cosines`` (odd spikes vs
    the even bank) -- NO new matcher run. Restricted to units with >= ``min_spikes_each_half`` spikes in
    BOTH halves; odd spikes subsampled to ``max_test_spikes`` per unit for cost. FAILS on strong drift
    across the parity split (a clean drifting unit under-claims its own held-out spikes).

    Returns ``{unit_id: {n_test, self_frac}}`` (self_frac = fraction of tested odd spikes whose best
    same-tetrode match in the EVEN bank is this unit).
    """
    rng = np.random.default_rng(seed)
    wlen = int(win_s * fs)
    groups = np.asarray(sorting.get_property("group"))
    gmap = {int(u): int(groups[i]) for i, u in enumerate(sorting.unit_ids)}
    even_trains, odd_trains = {}, {}
    for u in sorting.unit_ids:
        tr = np.sort(sorting.get_unit_spike_train(u)).astype(np.int64)
        parity = (tr // wlen) % 2
        ev, od = tr[parity == 0], tr[parity == 1]
        if ev.size >= min_spikes_each_half and od.size >= min_spikes_each_half:
            even_trains[int(u)] = ev
            odd_trains[int(u)] = od
    if not even_trains:
        return {}
    keep = sorted(even_trains)
    even_sorting = si.NumpySorting.from_unit_dict([{u: even_trains[u] for u in keep}], sampling_frequency=fs)
    even_sorting.set_property("group", np.array([gmap[u] for u in keep]))
    even_bank, _ = build_templates_object(even_sorting, recording, with_snr=False, n_jobs=n_jobs)
    bank_ids = [int(u) for u in even_bank.unit_ids]
    uid_to_idx = {u: i for i, u in enumerate(bank_ids)}
    # assemble odd test spikes with cluster_index pointing into the even bank
    samp, ci = [], []
    for u in bank_ids:
        od = odd_trains[u]
        if od.size > max_test_spikes:
            od = np.sort(rng.choice(od, size=max_test_spikes, replace=False))
        samp.append(od)
        ci.append(np.full(od.size, uid_to_idx[u], np.int64))
    samp = np.concatenate(samp)
    ci = np.concatenate(ci)
    odd_spikes = np.zeros(samp.size, dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    odd_spikes["sample_index"] = samp
    odd_spikes["cluster_index"] = ci
    a, b = asym_window_bounds(even_bank.nbefore)
    cos = all_template_cosines(recording, even_bank, odd_spikes, a, b, s_max)
    arg = cos["rF_arg"]
    out = {}
    for u in bank_ids:
        idx = uid_to_idx[u]
        sel = (ci == idx) & (arg >= 0)
        n = int(sel.sum())
        out[u] = dict(n_test=n, self_frac=float((arg[sel] == idx).mean()) if n else float("nan"))
    return out


# ---- combine the three signals ----------------------------------------------------------------

def assignment_purity_summary(cosine_purity, ccg_records, heldout=None, *,
                              cosine_frac_thresh=0.80, heldout_frac_thresh=0.80):
    """Combine the three internal purity signals per unit and flag impurity on >= 2-signal agreement.

    Each signal casts an IMPURITY vote that may ABSTAIN (None):
      * cosine  -- impure if ``best_match_frac < cosine_frac_thresh``; abstains if no finite spikes.
      * ccg     -- impure if the unit's top cosine neighbour is a DISTINCT cell (filled CCG -> real
                   cross-unit contamination); a 'duplicate' verdict votes NOT-impure (oversplit, an
                   axis-C merge issue, not B-contamination); ambiguous/segregated/no-neighbour abstain.
      * heldout -- impure if ``self_frac < heldout_frac_thresh``; abstains if not tested.
    ``flagged`` = (# impure votes) >= 2. ``category`` summarises WHY: 'cross_contaminated' (CCG distinct),
    'oversplit' (CCG duplicate but cosine impure -> merge candidate), 'flagged_ambiguous', or 'clean'.

    Returns ``{unit_id: {cosine_frac, cosine_vote, ccg_top_neighbor, ccg_verdict, ccg_vote, heldout_frac,
    heldout_vote, n_impure_votes, flagged, category}}`` -- every signal reported so disagreements are
    human-adjudicable. Thresholds and the >= 2 rule are provisional (calibrate per the plan).
    """
    # index CCG records by frozenset for order-independent lookup
    ccg_by_pair = {frozenset(k): v for k, v in ccg_records.items()}
    out = {}
    for uid, c in cosine_purity.items():
        cf = c["best_match_frac"]
        cosine_vote = None if not np.isfinite(cf) else (cf < cosine_frac_thresh)
        top = c["top_neighbor"]
        ccg_verdict = None
        if top is not None and top >= 0:
            rec = ccg_by_pair.get(frozenset((uid, top)))
            ccg_verdict = rec["verdict"] if rec else None
        if ccg_verdict == "distinct":
            ccg_vote = True
        elif ccg_verdict == "duplicate":
            ccg_vote = False
        else:
            ccg_vote = None  # ambiguous / SEGREGATED / no neighbour -> abstain
        hf = heldout.get(uid, {}).get("self_frac", np.nan) if heldout else np.nan
        heldout_vote = None if not np.isfinite(hf) else (hf < heldout_frac_thresh)
        votes = [v for v in (cosine_vote, ccg_vote, heldout_vote) if v is not None]
        n_impure = int(sum(bool(v) for v in votes))
        flagged = n_impure >= 2
        if flagged and ccg_verdict == "distinct":
            category = "cross_contaminated"
        elif flagged and ccg_verdict == "duplicate":
            category = "oversplit"
        elif flagged:
            category = "flagged_ambiguous"
        else:
            category = "clean"
        out[uid] = dict(cosine_frac=cf, cosine_vote=cosine_vote, ccg_top_neighbor=top,
                        ccg_verdict=ccg_verdict, ccg_vote=ccg_vote, heldout_frac=hf,
                        heldout_vote=heldout_vote, n_impure_votes=n_impure, flagged=flagged,
                        category=category)
    return out

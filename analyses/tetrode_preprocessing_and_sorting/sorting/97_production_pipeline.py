"""PRODUCTION pipeline: assemble the final tetrode MP deliverable and score it A/B/C over the whole 47 h.

Chains the validated components (scripts 90/87/92/86) into ONE checkpointed, resumable run, with the
parameters locked by the 2026-06-19 reorientation diagnostics:

  1. base   = ``assembled_reseed_c12`` (re-estimate + reseed carry-forward; the diagnostics' best base).
  2. MERGE  -> ``assembled_prod_merge``: CCG-guarded within-tetrode merge (cosine proposes >= 0.90; CCG
              'duplicate' accepts; cosine >= 0.95 fallback where CCG abstains). Lifts tight purity
              0.817 -> 0.872 and cuts CCG-duplicate pairs 72 -> 22 at zero coverage loss (merge_first json).
  3. RESIDUAL SUA -> ``assembled_prod_sua``: over EVERY window, the high-MAD (>= 10) events the merged base
              leaves unclaimed are PROPOSED to their best same-tetrode template (cosine >= 0.8) and folded
              into that host unit iff they do not break its +/-1.5 ms refractory (script-87 E2 recovered
              ~33% cleanly; 0 'distinct'/cross-contaminating CCG verdicts). This is the step the E2
              prototype only *measured* -- here it is persisted.
  4. MUA    -> ``assembled_prod``: over EVERY window, events still unclaimed and >= 7 MAD whose best
              same-tetrode cosine is in [theta=0.55, 0.8) are pooled into ONE is_mua=True pseudo-unit per
              tetrode (neural-but-unsortable); < theta = noise (dropped); >= 0.8 = left for SUA. SUA units
              carry is_mua=False. (The on-disk ``assembled_mergefirst_mua`` used only 3 windows; this covers
              all of them so the MUA train spans the recording.)
  5. SCORE  A/B/C over all ~94 windows for base vs the SUA deliverable, + axis-A coverage for SUA+MUA.

Order matters: SUA -> SUA-residual -> MUA, so the MUA bucket never eats a recoverable SUA spike (each
stage recomputes the claimed mask against the augmented base). Every stage persists a sorting and skips if
it already exists (``--force`` rebuilds), so a crash resumes from the last completed stage.

    cd gfys_workspace
    # smoke (2 windows, throwaway tag) then the real run:
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/97_production_pipeline.py \
        --tag _smoke --max-windows 2
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/97_production_pipeline.py
"""
import argparse
import json
import pathlib
import shutil
import time

import numpy as np
import spikeinterface as si

from _assignment_eval import best_template_for_events, ccg_guarded_merge, ccg_verdict_pair, window_bank
from _mp_common import FS, materialize_span
from _scoreboard import (axis_a_summary, axis_b_aggregate, axis_c_summary, compare_scoreboards,
                         windowed_axis_b)
from _wobble_eval import coverage_by_band

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
# locked parameters (2026-06-19 reorientation diagnostics)
PROPOSE_COS, FALLBACK_COS = 0.90, 0.95      # CCG-guarded merge
RESIDUAL_MAD_FLOOR, RESIDUAL_TAU = 10.0, 0.8  # residual SUA capture
MUA_THETA_DEFAULT, MUA_MAD_FLOOR, SUA_TAU = 0.55, 7.0, 0.8  # MUA bucket
REFR_MS, COINCIDENCE_MS = 1.5, 0.3
MIN_SPIKES_TEMPLATE = 100                   # template-reliability floor (script-93 sweep-stable)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refractory_violation(event_frames, host_train, refr_frames):
    """Bool per event: a host spike within +/-refr_frames (inserting the event would break refractoriness)."""
    if host_train.size == 0:
        return np.zeros(event_frames.size, dtype=bool)
    h = np.sort(host_train)
    j = np.searchsorted(h, event_frames)
    dprev = np.where(j > 0, event_frames - h[np.clip(j - 1, 0, h.size - 1)], refr_frames + 1)
    dnext = np.where(j < h.size, h[np.clip(j, 0, h.size - 1)] - event_frames, refr_frames + 1)
    return np.minimum(dprev, dnext) <= refr_frames


def collapse_coincident(frames, tol):
    """Sorted frames with within-tol coincidences collapsed to their first occurrence."""
    if frames.size == 0 or tol <= 0:
        return frames
    s = np.sort(frames)
    return s[np.concatenate([[True], np.diff(s) > tol])]


def window_starts(nfr, *, max_windows=None):
    """Non-overlapping 1800 s window start frames covering [0, nfr)."""
    wf = int(WIN_S * FS)
    starts = list(range(0, nfr, wf))
    return starts if max_windows is None else starts[:max_windows]


def load_peaks():
    """(peak_s, peak_g, amp_mad) globally-sorted detected peaks from the shared spike_coverage.npz."""
    z = np.load(OUT / "spike_coverage.npz", mmap_mode="r")
    return np.asarray(z["peak_sample"]), np.asarray(z["peak_group"]), np.asarray(z["amp_mad"])


# ----------------------------------------------------------------------------- stage 2: merge
def stage_merge(rec, base_name, out_name, *, force):
    out_dir = OUT / out_name
    if out_dir.exists() and not force:
        m = si.load(out_dir)
        _log(f"merge: reuse {out_name} ({m.get_num_units()} units)")
        return m
    base = si.load(OUT / base_name)
    _log(f"merge: {base_name} ({base.get_num_units()}u) -> CCG-guarded merge "
         f"(propose>={PROPOSE_COS}, fallback>={FALLBACK_COS})")
    merged, merges = ccg_guarded_merge(base, rec, propose_cos=PROPOSE_COS, fallback_cos=FALLBACK_COS,
                                       win_s=WIN_S, n_jobs=N_JOBS)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    merged.save(folder=out_dir)
    _log(f"merge: {base.get_num_units()} -> {merged.get_num_units()} units ({len(merges)} pair-merges) "
         f"-> {out_name}")
    return merged


# ----------------------------------------------------------- stage 3: residual SUA capture (persist)
def stage_residual(rec, merged, peaks, starts, out_name, *, force):
    out_dir = OUT / out_name
    if out_dir.exists() and not force:
        s = si.load(out_dir)
        _log(f"residual: reuse {out_name} ({s.get_num_units()} units)")
        return s
    peak_s, peak_g, amp_mad = peaks
    nfr = rec.get_num_frames()
    refr_frames = int(REFR_MS * 1e-3 * FS)
    tol = int(COINCIDENCE_MS * 1e-3 * FS)
    grp_of = {int(u): int(g) for u, g in zip(merged.unit_ids, np.asarray(merged.get_property("group")))}
    orig_trains = {int(u): np.sort(merged.get_unit_spike_train(u).astype(np.int64)) for u in merged.unit_ids}

    # residual = high-MAD events the merged base leaves unclaimed (pooled-per-tetrode coverage)
    _, claimed = coverage_by_band(merged, peak_s, peak_g, amp_mad)
    resid_mask = (~claimed) & (amp_mad >= RESIDUAL_MAD_FLOOR)
    _log(f"residual: {int(resid_mask.sum()):,} unclaimed >= {RESIDUAL_MAD_FLOOR} MAD events over the "
         f"recording; {len(starts)} windows, tau_cos {RESIDUAL_TAU}")

    recovered = {}                              # host uid -> list of absolute recovered frames
    tally = dict(residual=0, no_template=0, refractory_reject=0, recovered=0)
    wf = int(WIN_S * FS)
    for wi, a in enumerate(starts):
        b = min(a + wf, nfr)
        lo, hi = np.searchsorted(peak_s, [a, b])
        sel = resid_mask[lo:hi]
        if not sel.any():
            continue
        win, bank, trains = window_bank(rec, merged, a, b, n_jobs=N_JOBS,
                                        min_spikes_template=MIN_SPIKES_TEMPLATE)
        if bank is None:
            continue
        es = (peak_s[lo:hi][sel] - a).astype(np.int64)
        eg = peak_g[lo:hi][sel].astype(np.int64)
        tally["residual"] += int(es.size)
        best_cos, best_uid, valid = best_template_for_events(win, bank, es, eg)
        propose = valid & (best_cos >= RESIDUAL_TAU)
        tally["no_template"] += int((~propose).sum())
        for uid in np.unique(best_uid[propose]):
            idx = np.flatnonzero(propose & (best_uid == uid))
            ok = ~refractory_violation(es[idx], trains[int(uid)], refr_frames)
            tally["recovered"] += int(ok.sum())
            tally["refractory_reject"] += int((~ok).sum())
            recovered.setdefault(int(uid), []).extend((es[idx[ok]] + a).tolist())
        if (wi + 1) % 10 == 0:
            _log(f"residual: window {wi + 1}/{len(starts)} | recovered so far {tally['recovered']:,}")

    # fold recovered spikes into their host units; coincidence-collapse the augmented train
    new_trains = {}
    for u, orig in orig_trains.items():
        extra = np.asarray(recovered.get(u, []), dtype=np.int64)
        aug = np.concatenate([orig, extra]) if extra.size else orig
        new_trains[u] = collapse_coincident(aug, tol)
    sua = si.NumpySorting.from_unit_dict([new_trains], sampling_frequency=FS)
    sua.set_property("group", np.asarray([grp_of[int(u)] for u in sua.unit_ids]))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    sua.save(folder=out_dir)

    # validate recovery quality per affected unit: CCG(recovered vs host original)
    vc = {}
    for u, frames in recovered.items():
        rec_fr = np.sort(np.asarray(frames, np.int64))
        if rec_fr.size < 5:
            continue
        v = ccg_verdict_pair(rec_fr, orig_trains[u], win_frames=wf)["verdict"]
        vc[v] = vc.get(v, 0) + 1
    n_r = max(tally["residual"], 1)
    _log(f"residual: recovered {tally['recovered']:,} ({100 * tally['recovered'] / n_r:.1f}%), "
         f"no-template {tally['no_template']:,} ({100 * tally['no_template'] / n_r:.1f}%), "
         f"refractory-reject {tally['refractory_reject']:,}; {len(recovered)} units gained spikes")
    _log(f"residual: recovered-vs-host CCG {vc} (duplicate=own dropout/clean; distinct=contaminant)")
    (OUT / f"production_residual{_tag_of(out_name)}.json").write_text(json.dumps(
        {"base": "merge", "mad_floor": RESIDUAL_MAD_FLOOR, "tau_cos": RESIDUAL_TAU,
         "tally": tally, "n_units_gained": len(recovered), "ccg_verdicts": vc}, indent=2))
    return sua


# ------------------------------------------------------------------ stage 4: per-tetrode MUA bucket
def stage_mua(rec, sua, peaks, starts, theta, out_name, *, force):
    out_dir = OUT / out_name
    if out_dir.exists() and not force:
        s = si.load(out_dir)
        _log(f"mua: reuse {out_name} ({s.get_num_units()} units)")
        return s
    peak_s, peak_g, amp_mad = peaks
    nfr = rec.get_num_frames()
    grp_of = {int(u): int(g) for u, g in zip(sua.unit_ids, np.asarray(sua.get_property("group")))}
    _, claimed = coverage_by_band(sua, peak_s, peak_g, amp_mad)
    mua_input = (~claimed) & (amp_mad >= MUA_MAD_FLOOR)
    _log(f"mua: theta {theta}, MAD floor {MUA_MAD_FLOOR}; {int(mua_input.sum()):,} unclaimed >= "
         f"{MUA_MAD_FLOOR} MAD events; {len(starts)} windows")

    mua_frames, n = {}, dict(classified=0, mua=0, noise=0, sua_recoverable=0)
    wf = int(WIN_S * FS)
    for wi, a in enumerate(starts):
        b = min(a + wf, nfr)
        lo, hi = np.searchsorted(peak_s, [a, b])
        sel = mua_input[lo:hi]
        if not sel.any():
            continue
        win, bank, _ = window_bank(rec, sua, a, b, n_jobs=N_JOBS, min_spikes_template=MIN_SPIKES_TEMPLATE)
        if bank is None:
            continue
        es = (peak_s[lo:hi][sel] - a).astype(np.int64)
        eg = peak_g[lo:hi][sel].astype(np.int64)
        best_cos, _, valid = best_template_for_events(win, bank, es, eg)
        is_mua = valid & (best_cos >= theta) & (best_cos < SUA_TAU)
        n["classified"] += int(valid.sum())
        n["mua"] += int(is_mua.sum())
        n["noise"] += int((valid & (best_cos < theta)).sum())
        n["sua_recoverable"] += int((valid & (best_cos >= SUA_TAU)).sum())
        for g in np.unique(eg[is_mua]):
            mua_frames.setdefault(int(g), []).append(es[is_mua & (eg == g)] + a)
        if (wi + 1) % 10 == 0:
            _log(f"mua: window {wi + 1}/{len(starts)} | MUA events so far {n['mua']:,}")

    tol = int(COINCIDENCE_MS * 1e-3 * FS)
    trains, groups, ismua = {}, [], []
    for u in sua.unit_ids:
        trains[int(u)] = np.sort(sua.get_unit_spike_train(u).astype(np.int64))
        groups.append(grp_of[int(u)])
        ismua.append(False)
    next_id = max(int(u) for u in sua.unit_ids) + 1
    mua_counts = {}
    for g, chunks in sorted(mua_frames.items()):
        allspk = collapse_coincident(np.concatenate(chunks), tol)
        trains[next_id] = allspk
        groups.append(g)
        ismua.append(True)
        mua_counts[g] = int(allspk.size)
        next_id += 1
    prod = si.NumpySorting.from_unit_dict([trains], sampling_frequency=FS)
    prod.set_property("group", np.asarray(groups))
    prod.set_property("is_mua", np.asarray(ismua))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    prod.save(folder=out_dir)
    c = max(n["classified"], 1)
    _log(f"mua: {n['classified']:,} classified -> MUA {n['mua']:,} ({100 * n['mua'] / c:.1f}%), "
         f"noise {n['noise']:,} ({100 * n['noise'] / c:.1f}%), sua-rec {n['sua_recoverable']:,}; "
         f"{len(mua_counts)} per-tetrode MUA units")
    (OUT / f"production_mua{_tag_of(out_name)}.json").write_text(json.dumps(
        {"theta": theta, "mad_floor": MUA_MAD_FLOOR, "counts": n, "n_mua_units": len(mua_counts),
         "mua_events_per_tetrode": mua_counts}, indent=2))
    return prod


# --------------------------------------------------------------------------------- stage 5: scoring
def _score_one(rec, sorting, peaks, windows_h, *, with_purity, with_identity=True):
    res = {"label": None, "n_units": int(sorting.get_num_units())}
    res["axis_A"] = axis_a_summary(sorting, peaks)
    # axis C is meaningless once the pooled per-tetrode MUA bucket is present (it is CCG-'duplicate' vs every
    # SUA unit on its tetrode by construction); report it only on the SUA unit set.
    if with_identity:
        res["axis_C"] = axis_c_summary(sorting, win_s=WIN_S)
    if with_purity:
        cp_full, cp_tight = windowed_axis_b(rec, sorting, windows_h, win_s=WIN_S, n_jobs=N_JOBS,
                                            min_spikes_template=MIN_SPIKES_TEMPLATE)
        b = axis_b_aggregate(cp_full, sorting, win_s=WIN_S)
        num = sum(c["n_finite"] * c["best_match_frac"] for c in cp_tight.values()
                  if np.isfinite(c["best_match_frac"]))
        den = sum(c["n_finite"] for c in cp_tight.values() if np.isfinite(c["best_match_frac"]))
        b["spikeweighted_purity_tight"] = float(num / den) if den else float("nan")
        res["axis_B"] = b
    return res


def stage_score(rec, variants, peaks, starts, *, tag):
    """variants: list of (label, sorting, with_purity, with_identity). Persist each as it finishes."""
    windows_h = [a / (3600.0 * FS) for a in starts]
    out_json = OUT / f"production_scoreboard{tag}.json"
    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    for label, sorting, with_purity, with_identity in variants:
        if label in done:
            _log(f"score: reuse {label}")
            continue
        _log(f"score: {label} ({sorting.get_num_units()}u) over {len(starts)} windows "
             f"(purity={'yes' if with_purity else 'coverage-only'})")
        res = _score_one(rec, sorting, peaks, windows_h, with_purity=with_purity,
                         with_identity=with_identity)
        res["label"] = label
        done[label] = res
        out_json.write_text(json.dumps(done, indent=2))
        _log(f"score: {label} done -> {out_json.name}")
    print("\n" + compare_scoreboards({k: done[k] for k in done}), flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="assembled_reseed_c12")
    ap.add_argument("--theta", type=float, default=None, help="MUA cosine floor (default E1 0.55)")
    ap.add_argument("--tag", default="", help="suffix for output sortings/JSON (smoke runs use _smoke)")
    ap.add_argument("--max-windows", type=int, default=None, help="limit windows (smoke test)")
    ap.add_argument("--no-score-base", action="store_true", help="skip the all-windows base re-score")
    ap.add_argument("--force", action="store_true", help="rebuild stages even if their output exists")
    args = ap.parse_args()
    theta = args.theta if args.theta is not None else MUA_THETA_DEFAULT
    t = args.tag

    t0 = time.time()
    _log(f"PRODUCTION pipeline: base={args.base}, theta={theta}, tag={t!r}, "
         f"max_windows={args.max_windows}")
    rec = materialize_span(OUT, START_S, DUR_S)
    peaks = load_peaks()
    starts = window_starts(rec.get_num_frames(), max_windows=args.max_windows)
    _log(f"recording {rec.get_num_frames() / FS / 3600:.1f}h -> {len(starts)} windows")

    merged = stage_merge(rec, args.base, f"assembled_prod_merge{t}", force=args.force)
    sua = stage_residual(rec, merged, peaks, starts, f"assembled_prod_sua{t}", force=args.force)
    prod = stage_mua(rec, sua, peaks, starts, theta, f"assembled_prod{t}", force=args.force)

    variants = [("prod_sua", sua, True, True), ("prod_sua+mua", prod, False, False)]
    if not args.no_score_base:
        variants.insert(0, ("base", si.load(OUT / args.base), True, True))
    stage_score(rec, variants, peaks, starts, tag=t)
    _log(f"DONE in {(time.time() - t0) / 3600:.2f}h | final deliverable: assembled_prod{t} "
         f"({prod.get_num_units()} units = {sua.get_num_units()} SUA + per-tetrode MUA)")


def _tag_of(out_name):
    """Recover the --tag suffix from an output sorting name (assembled_prod_sua_smoke -> _smoke)."""
    for stem in ("assembled_prod_sua", "assembled_prod_merge", "assembled_prod"):
        if out_name.startswith(stem):
            return out_name[len(stem):]
    return ""


if __name__ == "__main__":
    main()

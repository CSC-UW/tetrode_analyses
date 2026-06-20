"""RESIDUAL_CAPTURE_PLAN E2 prototype (measure-first): can high-MAD UNCLAIMED events be recovered cleanly?

Scope: the high-MAD events the reseed deliverable leaves UNCLAIMED (claimed_0 from spike_coverage.npz;
bank = assembled_reseed_rs, the sorting that produced that mask). The plan hypothesises ~50-75% of these
are within-unit DROPOUT recoverable by "cosine proposes, refractory disposes". This treats that as a
HYPOTHESIS TO MEASURE, not assert -- the supporting evidence (locally-exclusive peaks, MS5 resort,
template similarity) is NOT ground truth (per the plan + the threshold-derivation policy).

Per window: build window-local templates (window_bank); for each unclaimed event >= MAD floor on a
tetrode, PROPOSE its best same-tetrode template (best_template_for_events, full-window cosine), accept iff
cosine >= tau_cos AND inserting it does not fall within the refractory window of an existing spike of that
unit (refractory DISPOSE). VALIDATE per affected unit: pool recovered spikes, run the CCG arbiter vs the
unit's ORIGINAL train -- a refractory DIP ('duplicate' verdict) is consistent with the unit's OWN dropout
(clean recovery); a FILLED CCG ('distinct') means the recovered spikes are an independent co-located cell
(cross-unit contamination, NOT recovery). Also report rp_contamination before vs after per unit.

Outputs the recovered / no-template / refractory-rejected breakdown, the >= MAD-floor coverage gain over
the processed windows, and the per-unit contamination cost -- the evidence to decide whether Part A is
worth wiring into the windowed pass.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/87_sua_residual_capture_e2.py \
        [--windows-h 5 15 26 35 44] [--mad-floor 10] [--tau-cos 0.8]
"""
import argparse
import json
import pathlib
from collections import Counter, defaultdict

import numpy as np
import spikeinterface as si

from _assignment_eval import best_template_for_events, ccg_verdict_pair, window_bank
from _mp_common import FS, materialize_span
from _wobble_eval import rp_contam

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
BANK_NAME = "assembled_reseed_rs"   # the sorting whose claimed_0 mask we are augmenting
REFR_MS = 1.5


def refractory_violation(event_frames, host_train, refr_frames):
    """Bool per event: a host spike within +/-refr_frames (inserting the event would break refractoriness)."""
    if host_train.size == 0:
        return np.zeros(event_frames.size, dtype=bool)
    h = np.sort(host_train)
    j = np.searchsorted(h, event_frames)
    dprev = np.where(j > 0, event_frames - h[np.clip(j - 1, 0, h.size - 1)], refr_frames + 1)
    dnext = np.where(j < h.size, h[np.clip(j, 0, h.size - 1)] - event_frames, refr_frames + 1)
    return np.minimum(dprev, dnext) <= refr_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 15.0, 26.0, 35.0, 44.0])
    ap.add_argument("--mad-floor", type=float, default=10.0)
    ap.add_argument("--tau-cos", type=float, default=0.8)
    args = ap.parse_args()
    refr_frames = int(REFR_MS * 1e-3 * FS)
    win_frames = int(WIN_S * FS)

    rec = materialize_span(OUT, START_S, DUR_S)
    nfr = rec.get_num_frames()
    sorting = si.load(OUT / BANK_NAME)
    z = np.load(OUT / "spike_coverage.npz", mmap_mode="r")
    peak_sample = np.asarray(z["peak_sample"])          # globally sorted (per-chunk detect, concatenated)
    peak_group = z["peak_group"]
    amp_mad = z["amp_mad"]
    claimed = z["claimed_0"]                            # reseed deliverable (== assembled_reseed_rs)
    print(f"bank {BANK_NAME}: {sorting.get_num_units()} units; recording {nfr / FS / 3600:.1f}h; "
          f"MAD floor {args.mad_floor}, tau_cos {args.tau_cos}; windows {args.windows_h}h", flush=True)

    tally = Counter()
    recovered_abs = defaultdict(list)
    mad_recovered, mad_resid = [], []
    for h in args.windows_h:
        a = int(h * 3600 * FS)
        b = min(a + win_frames, nfr)
        if a >= nfr:
            print(f"  window {h}h beyond recording -- skip", flush=True)
            continue
        win, bank, trains = window_bank(rec, sorting, a, b, n_jobs=N_JOBS)
        if bank is None:
            print(f"  window {h}h: no unit >= template floor -- skip", flush=True)
            continue
        lo, hi = np.searchsorted(peak_sample, [a, b])
        es = peak_sample[lo:hi].astype(np.int64) - a    # window-local event frames
        eg = peak_group[lo:hi].astype(np.int64)
        emad = np.asarray(amp_mad[lo:hi])
        resid = (~np.asarray(claimed[lo:hi])) & (emad >= args.mad_floor)
        n_resid = int(resid.sum())
        tally["residual"] += n_resid
        mad_resid.extend(emad[resid].tolist())
        if n_resid == 0:
            print(f"  window {h}h: 0 residual >= {args.mad_floor} MAD events", flush=True)
            continue
        es_r, eg_r, mad_r = es[resid], eg[resid], emad[resid]
        best_cos, best_uid, valid = best_template_for_events(win, bank, es_r, eg_r)
        propose = valid & (best_cos >= args.tau_cos)
        accept = np.zeros(propose.size, dtype=bool)
        for uid in np.unique(best_uid[propose]):
            sel = np.flatnonzero(propose & (best_uid == uid))
            ok = ~refractory_violation(es_r[sel], trains[int(uid)], refr_frames)
            accept[sel[ok]] = True
            recovered_abs[int(uid)].extend((es_r[sel[ok]] + a).tolist())
        tally["no_template"] += int((~propose).sum())
        tally["refractory_reject"] += int((propose & ~accept).sum())
        tally["recovered"] += int(accept.sum())
        mad_recovered.extend(mad_r[accept].tolist())
        print(f"  window {h}h: {n_resid:,} residual -> recovered {int(accept.sum()):,} "
              f"({100 * accept.mean():.1f}%), no-template {int((~propose).sum()):,}, "
              f"refractory-reject {int((propose & ~accept).sum()):,}", flush=True)

    # per affected unit: CCG(recovered vs host) + rp before/after over the full recording
    unit_rows = []
    for uid, frames in sorted(recovered_abs.items()):
        orig = np.sort(sorting.get_unit_spike_train(uid).astype(np.int64))
        rec_fr = np.sort(np.array(frames, dtype=np.int64))
        aug = np.sort(np.concatenate([orig, rec_fr]))
        v = ccg_verdict_pair(rec_fr, orig, win_frames=win_frames)
        unit_rows.append(dict(uid=int(uid), n_recovered=int(rec_fr.size), n_orig=int(orig.size),
                              rp_before=float(rp_contam(orig, nfr)), rp_after=float(rp_contam(aug, nfr)),
                              ccg_verdict=v["verdict"], ccg_ratio=v["ratio"], n_co=v["n_co"]))

    n_resid = tally["residual"]
    print("\n" + "=" * 78)
    print(f"RESIDUAL-CAPTURE E2: {n_resid:,} residual >= {args.mad_floor} MAD events over "
          f"{len([h for h in args.windows_h])} windows")
    print("=" * 78)
    if n_resid:
        print(f"  recovered           {tally['recovered']:>9,} ({100 * tally['recovered'] / n_resid:5.1f}%)  "
              f"= coverage gain on the >= {args.mad_floor} MAD residual (over processed windows)")
        print(f"  no-template (cos<tau){tally['no_template']:>8,} ({100 * tally['no_template'] / n_resid:5.1f}%)  "
              f"= collision / MUA (no shape match)")
        print(f"  refractory-reject   {tally['refractory_reject']:>9,} "
              f"({100 * tally['refractory_reject'] / n_resid:5.1f}%)  = shape match but breaks host refractory")
    # CCG validation of recovery quality
    vc = Counter(r["ccg_verdict"] for r in unit_rows)
    print(f"\n  affected units: {len(unit_rows)}  | recovered-vs-host CCG: "
          + ", ".join(f"{k}={v}" for k, v in sorted(vc.items()))
          + "  (duplicate/dip = unit's OWN dropout = clean; distinct/filled = co-located CONTAMINANT)")
    if unit_rows:
        drp = np.array([r["rp_after"] - r["rp_before"] for r in unit_rows], dtype=float)
        drp = drp[np.isfinite(drp)]
        worst = sorted(unit_rows, key=lambda r: -(r["rp_after"] - r["rp_before"]
                                                  if np.isfinite(r["rp_after"]) else -1))[:8]
        print(f"  rp_contamination delta (after-before): median {np.median(drp):+.4f}, "
              f"max {np.max(drp):+.4f} over {drp.size} units")
        print("  largest contamination increases (uid: n_rec rp_before->rp_after ccg):")
        for r in worst:
            print(f"    u{r['uid']:<4} n_rec={r['n_recovered']:<6} {r['rp_before']:.3f}->{r['rp_after']:.3f}"
                  f"  ccg={r['ccg_verdict']}", flush=True)

    np.savez(OUT / "residual_capture_e2.npz",
             tally=np.array([(k, tally[k]) for k in tally], dtype=object),
             mad_recovered=np.array(mad_recovered), mad_resid=np.array(mad_resid),
             unit_uid=np.array([r["uid"] for r in unit_rows]),
             unit_n_recovered=np.array([r["n_recovered"] for r in unit_rows]),
             unit_rp_before=np.array([r["rp_before"] for r in unit_rows]),
             unit_rp_after=np.array([r["rp_after"] for r in unit_rows]),
             unit_ccg=np.array([r["ccg_verdict"] for r in unit_rows]))
    (OUT / "residual_capture_e2.json").write_text(json.dumps(
        {"params": vars(args), "tally": dict(tally), "n_affected_units": len(unit_rows),
         "ccg_verdicts": dict(vc)}, indent=2))
    print(f"\nwrote {OUT / 'residual_capture_e2.npz'} and .json\nDONE", flush=True)


if __name__ == "__main__":
    main()

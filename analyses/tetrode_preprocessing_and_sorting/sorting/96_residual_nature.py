"""Are the low-cosine (ill-fitting) spikes COLLISIONS/noise, or a coherent MISSED unit we should cluster?

Two tests on the merge-first sorting in a window:
  A. COINCIDENCE -- for low-cos (best same-tet cosine < tau) vs clean (>= hi) spikes, the fraction within
     +/-0.5 ms of ANOTHER same-tetrode unit's spike. Collisions are near-coincident with a second unit;
     a missed unit's spikes are not. Also splits the low-cos set by amplitude (collisions skew high-MAD,
     noise tail low-MAD).
  B. RE-CLUSTER -- on the busiest tetrode, KMeans the low-cos spike waveforms (PCA) and report, per
     sub-cluster, internal coherence (member cosine to its mean), refractory contamination, and novelty
     (max cosine to an existing bank template). A coherent + clean + novel sub-cluster = a missed unit
     worth rescuing; an incoherent smear = collisions/noise (-> MUA, not new SUA). Clean control: the same
     on an equal sample of clean spikes (should cluster tightly).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/96_residual_nature.py \
        [--sorting assembled_mergefirst] [--window-h 26] [--tau 0.8] [--hi 0.9]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from _assignment_eval import window_bank
from _mp_common import (FS, TIGHT_SHIFT, _unit_groups_from_mask, all_template_cosines,
                        asym_window_bounds, materialize_span)
from _wobble_eval import rp_contam

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16


def coincidence_frac(samples, others_sorted, tol):
    """Fraction of `samples` within +/-tol of any frame in sorted `others_sorted`."""
    if samples.size == 0 or others_sorted.size == 0:
        return float("nan")
    j = np.searchsorted(others_sorted, samples)
    dprev = np.where(j > 0, samples - others_sorted[np.clip(j - 1, 0, others_sorted.size - 1)], tol + 1)
    dnext = np.where(j < others_sorted.size, others_sorted[np.clip(j, 0, others_sorted.size - 1)] - samples,
                     tol + 1)
    return float(np.mean(np.minimum(dprev, dnext) <= tol))


def cohere(wfs):
    """Mean cosine of each waveform to the set mean (flattened vectors)."""
    if wfs.shape[0] < 2:
        return float("nan")
    m = wfs.mean(0)
    mn = np.linalg.norm(m)
    if mn == 0:
        return float("nan")
    nr = np.linalg.norm(wfs, axis=1)
    ok = nr > 0
    return float(np.mean((wfs[ok] @ m) / (nr[ok] * mn)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sorting", default="assembled_mergefirst")
    ap.add_argument("--window-h", type=float, default=26.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-wf", type=int, default=6000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    tol = int(0.5e-3 * FS)

    rec = materialize_span(OUT, START_S, DUR_S)
    sorting = si.load(OUT / args.sorting)
    a = int(args.window_h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win, bank, trains = window_bank(rec, sorting, a, b, n_jobs=N_JOBS)
    if bank is None:
        print("empty window -- abort", flush=True)
        return
    bank_ids = np.asarray([int(u) for u in bank.unit_ids])
    samp, ci = [], []
    for i, u in enumerate(bank_ids):
        t = trains[int(u)]
        samp.append(t)
        ci.append(np.full(t.size, i, np.int64))
    sp = np.zeros(int(sum(len(s) for s in samp)), dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    sp["sample_index"] = np.concatenate(samp)
    sp["cluster_index"] = np.concatenate(ci)
    aa, bb = asym_window_bounds(bank.nbefore)
    cos = all_template_cosines(win, bank, sp, aa, bb, TIGHT_SHIFT)
    rec_groups = np.asarray(win.get_property("group"))
    chan_ids = np.asarray(win.channel_ids)
    ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
    nbefore, n_samp, nfr = bank.nbefore, np.asarray(bank.get_dense_templates()).shape[1], win.get_num_frames()
    rFb = cos["rF_best"]
    mad = cos["mad"]
    grp_of_spike = ug[sp["cluster_index"]]
    fin = np.isfinite(rFb)
    low = fin & (rFb < args.tau)
    high = fin & (rFb >= args.hi)
    print(f"{args.sorting} @ {args.window_h}h: {int(fin.sum()):,} spikes; "
          f"low-cos(<{args.tau}) {int(low.sum()):,} ({100*low.mean():.1f}%), clean(>={args.hi}) "
          f"{int(high.sum()):,}", flush=True)

    # ---- Test A: coincidence with OTHER same-tetrode units ----
    print("\n=== A. coincidence with another same-tetrode unit (collision signature) ===", flush=True)
    res_a = {}
    for name, mask in [("low-cos", low), ("clean", high)]:
        fracs = []
        for g in np.unique(grp_of_spike[mask]):
            on = mask & (grp_of_spike == g)
            units_g = np.flatnonzero(ug == g)
            for ui in units_g:
                s_u = sp["sample_index"][on & (sp["cluster_index"] == ui)]
                if s_u.size == 0:
                    continue
                others = np.sort(np.concatenate(
                    [trains[int(bank_ids[uj])] for uj in units_g if uj != ui] or [np.empty(0, np.int64)]))
                fracs.append((coincidence_frac(s_u.astype(np.int64), others, tol), s_u.size))
        fracs = [(fr, n) for fr, n in fracs if n > 0 and np.isfinite(fr)]  # drop zero-weight/NaN units
        if fracs:
            f = np.array([x[0] for x in fracs])
            w = np.array([x[1] for x in fracs])
            res_a[name] = float(np.sum(f * w) / w.sum())
    print(f"  coincident-with-another-unit fraction:  low-cos {res_a.get('low-cos', float('nan')):.3f}  "
          f"vs clean {res_a.get('clean', float('nan')):.3f}", flush=True)
    lo_mad = mad[low & np.isfinite(mad)]
    print(f"  low-cos amplitude: frac >=10 MAD {100*np.mean(lo_mad >= 10):.1f}% (collision-skew), "
          f"frac <7 MAD {100*np.mean(lo_mad < 7):.1f}% (noise tail); median {np.median(lo_mad):.1f} MAD",
          flush=True)

    # ---- Test B: re-cluster low-cos waveforms on the busiest tetrode ----
    print("\n=== B. re-cluster low-cos spikes (coherent missed unit, or smear?) ===", flush=True)
    busiest = int(np.bincount(grp_of_spike[low].astype(int)).argmax())
    chans = np.flatnonzero(rec_groups == busiest)
    dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
    units_g = np.flatnonzero(ug == busiest)
    bank_wf = dense[units_g][:, :, chans].reshape(len(units_g), -1)  # existing templates on this tetrode

    def waveforms(idx):
        out = []
        for i in idx:
            off = int(sp["sample_index"][i]) - nbefore
            if off < 0 or off + n_samp > nfr:
                continue
            tr = np.asarray(win.get_traces(start_frame=off, end_frame=off + n_samp,
                                           channel_ids=list(chan_ids[chans])), dtype=np.float32)
            out.append(tr.reshape(-1))
        return np.asarray(out)

    def cluster_report(name, mask):
        idx = np.flatnonzero(mask & (grp_of_spike == busiest))
        if idx.size > args.max_wf:
            idx = rng.choice(idx, args.max_wf, replace=False)
        wf = waveforms(idx)
        if wf.shape[0] < args.k * 10:
            print(f"  {name}: too few waveforms ({wf.shape[0]})", flush=True)
            return []
        pcs = PCA(n_components=min(10, wf.shape[1])).fit_transform(wf)
        lab = KMeans(n_clusters=args.k, n_init=5, random_state=0).fit_predict(pcs)
        rows = []
        for cl in range(args.k):
            m = lab == cl
            if m.sum() < 20:
                continue
            coh = cohere(wf[m])
            mean_wf = wf[m].mean(0)
            nov = float(np.max([(mean_wf @ t) / (np.linalg.norm(mean_wf) * np.linalg.norm(t) + 1e-9)
                                for t in bank_wf])) if len(bank_wf) else float("nan")
            st = np.sort(sp["sample_index"][idx[m]].astype(np.int64))
            rp = rp_contam(st, nfr)
            rows.append(dict(n=int(m.sum()), coherence=round(coh, 3), max_cos_to_bank=round(nov, 3),
                             rp=round(float(rp), 3) if np.isfinite(rp) else None))
        rows.sort(key=lambda r: -r["coherence"])
        print(f"  {name} (tet {busiest}, {wf.shape[0]} wf, k={args.k}): best sub-clusters by coherence:",
              flush=True)
        for r in rows[:4]:
            print(f"     n={r['n']:<5} coherence={r['coherence']:.3f}  max_cos_to_bank={r['max_cos_to_bank']}"
                  f"  rp={r['rp']}", flush=True)
        return rows

    low_rows = cluster_report("low-cos", low)
    high_rows = cluster_report("clean (control)", high)
    # a missed unit = coherent + novel + clean refractory
    missed = [r for r in low_rows if r["coherence"] >= 0.85 and (r["max_cos_to_bank"] or 1) < 0.8
              and (r["rp"] is not None and r["rp"] <= 0.2)]
    print(f"\n  low-cos sub-clusters meeting MISSED-UNIT bar (coh>=0.85, cos_to_bank<0.8, rp<=0.2): "
          f"{len(missed)}", flush=True)

    (OUT / "residual_nature.json").write_text(json.dumps({
        "window_h": args.window_h, "tau": args.tau, "hi": args.hi,
        "n_low": int(low.sum()), "n_clean": int(high.sum()),
        "coincidence_low": res_a.get("low-cos"), "coincidence_clean": res_a.get("clean"),
        "low_mad_frac_ge10": float(np.mean(lo_mad >= 10)), "low_mad_frac_lt7": float(np.mean(lo_mad < 7)),
        "busiest_tetrode": busiest, "low_subclusters": low_rows, "clean_subclusters": high_rows,
        "n_missed_unit_candidates": len(missed),
    }, indent=2))
    print(f"\nwrote {OUT / 'residual_nature.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""Pick wobble's threshold on its OWN merits (no circus reference): the refractory-contamination knee.

Finer sweep around 0.5-0.9x ||t||^2 median on h=26. For each threshold (deduped wobble output) report:
  * median rp_contamination (>=50-spk units) + the per-unit count with rp_contamination < 0.1
    (BombCell's default max-tolerable rp for a GOOD unit, bombcell_curation.py:69 -- the SAME Llobet
    metric we compute), and isolation-tier counts (permissive/moderate/conservative),
  * >=12 MAD coverage + spurious fraction,
  * MARGINAL-spike quality: of the spikes newly admitted when stepping DOWN from the next-higher
    threshold, what fraction land on a real >=5.5 MAD detected peak (supported) -- i.e. is lowering the
    threshold buying real events or junk?
The knee = the lowest threshold before contamination breaks upward / marginal additions stop being real.

NOTE: only BombCell's rp_contamination<0.1 bound is used (geometry-free, directly comparable). BombCell's
shape/exp_decay metrics are geometry-dependent (exp_decay = spatial decay across channels) and ill-defined
on 4 co-located fictional-geometry tetrode wires, so full bombcell_label_units is intentionally NOT run.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes \
        python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/73_wobble_threshold_intrinsic.py [26]
"""
import json
import pathlib
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _mp_common import (_unit_groups_from_mask, build_templates_object, dedup_sorting,
                        materialize_span, run_matching, wobble_method_kwargs)
from _track_eval import TIERS, isolation_tier_mask
from _wobble_eval import (_by_tet, _within_tol, coverage_by_band, detect_window_peaks,
                          precision_summary, quality_df, spurious_fraction, tsq_median)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
TOL = int(0.5e-3 * FS)
BOMBCELL_RP_MAX = 0.1  # bombcell_curation.py:69 -- rp_contamination >= 0.1 -> downgraded from "good" to MUA
GRID_FACTORS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]


def main():
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    a = int(h * 3600 * FS)
    b = min(a + int(WIN_S * FS), rec.get_num_frames())
    win = rec.frame_slice(a, b)
    win.reset_times()

    sdir = WV / "intrinsic" / f"w{int(h)}h_ref"
    shutil.rmtree(sdir, ignore_errors=True)
    t0 = time.perf_counter()
    ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
    bank, _ = build_templates_object(ms5, win, with_snr=True, n_jobs=N_JOBS, seed=0)
    ug_map = {int(u): int(g) for u, g in
              zip(bank.unit_ids, _unit_groups_from_mask(bank.sparsity.mask, rec_groups))}
    med = tsq_median(bank)
    peak_s, peak_g, amp_mad = detect_window_peaks(win, n_jobs=N_JOBS)
    pbt = {int(g): np.sort(peak_s[peak_g == g]) for g in np.unique(peak_g)}
    print(f"window @ {h:.0f}h: {bank.unit_ids.size} units, ||t||^2 median={med:.3g}, "
          f"{peak_s.size:,} events ({(amp_mad>=12).sum():,} >=12 MAD), setup {time.perf_counter()-t0:.0f}s",
          flush=True)
    print(f"BombCell good-unit bound: rp_contamination < {BOMBCELL_RP_MAX}\n", flush=True)

    runs, per_tet = [], {}
    for f in GRID_FACTORS:
        thr = f * med
        t0 = time.perf_counter()
        mp, spikes = run_matching(win, bank, method="wobble",
                                  method_kwargs=wobble_method_kwargs(bank, threshold=thr), n_jobs=N_JOBS)
        mp.set_property("group", np.array([ug_map[int(u)] for u in mp.unit_ids]))
        dd = dedup_sorting(mp, win)
        qm = quality_df(dd, win, n_jobs=N_JOBS, with_amplitude_cutoff=True)
        prec = precision_summary(dd, qm)
        rp = np.asarray(qm["rp_contamination"], dtype=float)
        ac = np.asarray(qm["amplitude_cutoff"], dtype=float)
        nspk = np.array([dd.get_unit_spike_train(u).size for u in qm.index])
        evalmask = np.isfinite(rp) & (nspk >= 50)
        rp_good = int(np.sum(evalmask & (rp < BOMBCELL_RP_MAX)))
        n_eval = int(np.sum(evalmask))
        med_ac = float(np.median(ac[evalmask & np.isfinite(ac)])) if np.any(evalmask & np.isfinite(ac)) else float("nan")
        cov, _ = coverage_by_band(dd, peak_s, peak_g, amp_mad)
        spur = spurious_fraction(dd, peak_s, peak_g)
        tiers = {t: int(isolation_tier_mask(qm, t).sum()) for t in TIERS}
        per_tet[f] = _by_tet(dd)
        runs.append({"factor": f, "threshold": thr, "n_spikes": int(spikes.size),
                     "n_units": int(dd.get_num_units()), "median_rp": prec["median_rp_contamination"],
                     "median_amplitude_cutoff": med_ac, "rp_good_lt0.1": rp_good, "n_rp_evaluable": n_eval,
                     "tier_counts": tiers, "cov12": cov[">=12_pooled"], "spurious": float(spur)})
        print(f"  f={f:.2f} thr={thr:.4g}: {spikes.size:>8,}sp {dd.get_num_units():>3}u | med_rp "
              f"{prec['median_rp_contamination']:.4f} | amp_cutoff {med_ac:.3f} | rp<0.1 {rp_good}/{n_eval} | "
              f"tiers P{tiers['permissive']}/M{tiers['moderate']}/C{tiers['conservative']} | >=12 "
              f"{cov['>=12_pooled']:.1f}% | spur {spur*100:.1f}% | {time.perf_counter()-t0:.0f}s", flush=True)

    # marginal-spike quality: stepping DOWN from the next-higher factor, are the newly-admitted spikes real?
    facs = sorted(per_tet)
    for i in range(len(facs) - 1):
        f_lo, f_hi = facs[i], facs[i + 1]  # f_lo = looser (lower threshold, more spikes)
        lo, hi = per_tet[f_lo], per_tet[f_hi]
        newly_s, newly_g = [], []
        for g, st_lo in lo.items():
            st_hi = hi.get(g, np.empty(0, np.int64))
            matched = _within_tol(st_lo, np.full(st_lo.size, g), {g: st_hi}, TOL)
            newly_s.append(st_lo[~matched])
            newly_g.append(np.full(int((~matched).sum()), g))
        ns = np.concatenate(newly_s) if newly_s else np.empty(0, np.int64)
        ng = np.concatenate(newly_g) if newly_g else np.empty(0, np.int64)
        supp = _within_tol(ns, ng, pbt, TOL) if ns.size else np.empty(0, bool)
        marg = float(supp.mean()) if ns.size else float("nan")
        runs[i]["marginal_added_vs_next"] = int(ns.size)
        runs[i]["marginal_supported_frac"] = marg
        print(f"  step {f_hi:.2f}->{f_lo:.2f}: +{ns.size:,} spikes, {marg*100:.0f}% land on a >=5.5 MAD peak",
              flush=True)

    # knee candidates
    asc = sorted(runs, key=lambda r: r["factor"])
    bombcell_knee = next((r["factor"] for r in asc
                          if np.isfinite(r["median_rp"]) and r["median_rp"] < BOMBCELL_RP_MAX), None)
    clean_knee = next((r["factor"] for r in asc
                       if np.isfinite(r["median_rp"]) and r["median_rp"] < 0.03), None)
    print(f"\nKNEE: BombCell-median (lowest f with median rp<0.1) = {bombcell_knee}; "
          f"clean (median rp<0.03) = {clean_knee}", flush=True)

    # figure
    fa = np.array([r["factor"] for r in asc])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(fa, [r["median_rp"] for r in asc], "o-", color="#3b7dd8",
               label="median rp_contamination (false +)")
    ax[0].plot(fa, [r["median_amplitude_cutoff"] for r in asc], "v-", color="#2e8b57",
               label="median amplitude_cutoff (false -)")
    ax[0].axhline(BOMBCELL_RP_MAX, color="#c0392b", ls="--", label=f"BombCell good bound ({BOMBCELL_RP_MAX})")
    ax[0].axhline(0.03, color="0.6", ls=":", label="clean (0.03)")
    if clean_knee:
        ax[0].axvline(clean_knee, color="0.4", ls=":", label=f"clean knee f={clean_knee}")
    ax[0].set_xlabel("threshold factor (x ||t||^2 median)")
    ax[0].set_ylabel("contamination / cutoff")
    ax[0].set_title(f"Two-sided knee @ {h:.0f}h: rp_contamination (rises) vs amplitude_cutoff (falls)")
    ax[0].legend(fontsize=8)
    ax[1].plot(fa, [r["cov12"] for r in asc], "o-", color="#2e8b57", label=">=12 MAD coverage %")
    ax[1].plot(fa, [r["spurious"] * 100 for r in asc], "s--", color="#7d3bd8", label="spurious %")
    ax[1].plot(fa, [100.0 * r["rp_good_lt0.1"] / max(1, r["n_rp_evaluable"]) for r in asc], "^-",
               color="#d8743b", label="% units rp<0.1 (BombCell-good)")
    mfa = [r["factor"] for r in asc if "marginal_supported_frac" in r]
    mvf = [r["marginal_supported_frac"] * 100 for r in asc if "marginal_supported_frac" in r]
    ax[1].plot(mfa, mvf, "d:", color="0.4", label="marginal added-spike supported %")
    ax[1].set_xlabel("threshold factor (x ||t||^2 median)")
    ax[1].set_ylabel("%")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("Coverage / over-detection / cleanliness vs threshold")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    figp = WV / f"intrinsic_knee_w{int(h)}h.png"
    fig.savefig(figp, dpi=130)
    plt.close(fig)

    out = {"window_h": h, "tsq_median": med, "bombcell_rp_max": BOMBCELL_RP_MAX,
           "bombcell_median_knee_factor": bombcell_knee, "clean_knee_factor": clean_knee, "runs": runs}
    (WV / f"intrinsic_knee_w{int(h)}h.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {figp}\nwrote {WV / f'intrinsic_knee_w{int(h)}h.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

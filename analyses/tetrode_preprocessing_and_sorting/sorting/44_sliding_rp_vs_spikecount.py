"""Does sliding_rp_violation penalize low-spike-count units (low power, not contamination)?

sliding_rp_violation is a 90%-confidence UPPER BOUND on contamination, so it depends on
statistical power (spike count). The worry: low-firing / short-span units get a HIGH value
because there aren't enough spikes to prove they're clean -- not because they're contaminated
-- which would cut against keeping low-FR L2/3 cells.

Test: among units that the POINT estimate rp_contamination calls clean (<= 0.1), does
sliding_rp_violation still flag them (> 0.1 or NaN) preferentially at LOW spike count? If yes,
the metric is power-penalizing here. Disentangles power (spike count) from true contamination
(rp point estimate). Reads only the analyzer's quality_metrics.
"""
import pathlib

import matplotlib
import numpy as np
import spikeinterface as si
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/tracked_48h")
ANALYZER = T / "analyzer_clustered.zarr"
PASS = 0.1  # conservative isolation threshold


def main():
    an = si.load_sorting_analyzer(str(ANALYZER))
    qm = an.get_extension("quality_metrics").get_data()
    srp = qm["sliding_rp_violation"].to_numpy()
    rp = qm["rp_contamination"].to_numpy()
    ns = qm["num_spikes"].to_numpy().astype(float)
    fr = qm["firing_rate"].to_numpy()

    srp_nan = ~np.isfinite(srp)
    print(f"total units={len(qm)}  num_spikes: median={np.median(ns):.0f} "
          f"p10={np.percentile(ns,10):.0f} p90={np.percentile(ns,90):.0f}", flush=True)
    print(f"firing_rate: median={np.median(fr):.3f} Hz  p10={np.percentile(fr,10):.3f} p90={np.percentile(fr,90):.2f}", flush=True)
    print(f"sliding_rp_violation NaN: {int(srp_nan.sum())}/{len(qm)} ({srp_nan.mean()*100:.1f}%)", flush=True)

    fin = np.isfinite(srp)
    rho_ns = spearmanr(np.log10(ns[fin]), srp[fin]).statistic
    rho_fr = spearmanr(np.log10(fr[fin]), srp[fin]).statistic
    print(f"Spearman(sliding_rp, log10 num_spikes) = {rho_ns:.2f}  "
          f"(negative => fewer spikes -> higher/worse sliding_rp)", flush=True)
    print(f"Spearman(sliding_rp, log10 firing_rate) = {rho_fr:.2f}", flush=True)

    # ---- disentangle: among rp-CLEAN units, does sliding_rp flag the low-count ones? ----
    rp_clean = (rp <= PASS) & np.isfinite(rp)
    print(f"\nrp_contamination-clean (<= {PASS}) units: {int(rp_clean.sum())}", flush=True)
    # spike-count quintiles over ALL units, applied to the rp-clean subset
    qedges = np.quantile(ns, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\nAmong rp-clean units, does sliding_rp agree?  (srp_fail = NaN or > 0.1)", flush=True)
    print(f"{'ns_bin':<22}{'n':>5}{'med_ns':>9}{'med_fr':>8}{'srp_nan%':>9}{'srp_pass%':>10}{'med_srp':>9}", flush=True)
    bins_x, pass_frac = [], []
    for i in range(5):
        lo, hi = qedges[i], qedges[i + 1]
        m = rp_clean & (ns >= lo) & (ns <= hi if i == 4 else ns < hi)
        if m.sum() == 0:
            continue
        srp_m = srp[m]
        nanf = (~np.isfinite(srp_m)).mean()
        passf = (np.isfinite(srp_m) & (srp_m <= PASS)).mean()
        medsrp = np.nanmedian(srp_m)
        print(f"[{lo:>7.0f},{hi:>8.0f}]{'':<3}{int(m.sum()):>5}{np.median(ns[m]):>9.0f}"
              f"{np.median(fr[m]):>8.2f}{nanf*100:>8.0f}%{passf*100:>9.0f}%{medsrp:>9.3f}", flush=True)
        bins_x.append(np.median(ns[m]))
        pass_frac.append(passf)

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
    # A: sliding_rp vs num_spikes, colored by rp-clean; NaN plotted at sentinel 0.40
    y = np.where(np.isfinite(srp), srp, 0.40)
    for mask, c, lab in [(~rp_clean, "tab:orange", "rp-dirty (>0.1)"), (rp_clean, "tab:blue", "rp-clean (≤0.1)")]:
        axA.scatter(ns[mask], y[mask], s=10, alpha=0.4, c=c, edgecolors="none", label=lab)
    axA.axhline(PASS, color="k", lw=1, ls="--", label="pass ≤0.1")
    axA.axhline(0.40, color="0.6", lw=0.8, ls=":", )
    axA.text(axA.get_xlim()[1], 0.405, " NaN (unbounded)", fontsize=7, va="bottom", ha="right", color="0.4")
    axA.set_xscale("log")
    axA.set_xlabel("num_spikes (log)")
    axA.set_ylabel("sliding_rp_violation  (NaN shown at 0.40)")
    axA.set_title(f"sliding_rp vs spike count  (Spearman ρ={rho_ns:.2f})", fontsize=10)
    axA.legend(fontsize=8)
    # B: among rp-clean units, fraction passing sliding_rp by spike-count bin
    axB.plot(bins_x, np.array(pass_frac) * 100, "o-", color="tab:blue")
    axB.set_xscale("log")
    axB.set_ylim(0, 105)
    axB.set_xlabel("num_spikes (bin median, log)")
    axB.set_ylabel("% of rp-clean units that also pass sliding_rp ≤0.1")
    axB.set_title("Power penalty: do rp-clean units pass sliding_rp at low spike count?", fontsize=10)
    axB.grid(alpha=0.3)
    fig.suptitle("sliding_rp_violation vs statistical power (spike count) on the tracked units", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = T / "sliding_rp_vs_spikecount.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

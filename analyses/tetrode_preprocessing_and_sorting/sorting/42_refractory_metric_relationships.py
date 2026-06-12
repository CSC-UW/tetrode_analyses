"""How do the three refractory-contamination metrics relate on this tetrode data?

Decides whether the isolation tiers should AND or OR `isi_violations_ratio` and
`rp_contamination` (and where `sliding_rp_violation` sits). Plots, for the chunk-tracked
clustered analyzer:
  (A) sliding_rp_violation  vs  rp_contamination
  (B) rp_contamination      vs  isi_violations_ratio
as joint scatters with marginal histograms, with the tier thresholds (0.1 / 0.3 / 0.5 =
conservative / moderate / permissive) drawn on both axes. Prints the Spearman correlation
and, per threshold, the AND vs OR counts and the "disagreement" units (pass one metric but
not the other) -- the units whose tier membership depends on the AND/OR choice.

Reads only the analyzer's quality_metrics (no compute).
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
THRESH = {"conservative": 0.1, "moderate": 0.3, "permissive": 0.5}


def joint_panel(fig, gs_col, x, y, xlab, ylab, finite):
    """A scatter (main) + top/right marginal hists, threshold lines on both axes."""
    gs = gs_col.subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                            wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    xf, yf = x[finite], y[finite]
    ax.scatter(xf, yf, s=6, alpha=0.35, c="tab:blue", edgecolors="none")
    for t in THRESH.values():
        ax.axvline(t, color="0.6", lw=0.7, ls="--")
        ax.axhline(t, color="0.6", lw=0.7, ls="--")
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.02, 0.62)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    bins = np.linspace(0, 0.62, 40)
    axx.hist(np.clip(xf, 0, 0.62), bins=bins, color="tab:blue", alpha=0.7)
    axx.set_yscale("log")
    axx.tick_params(labelbottom=False)
    axy.hist(np.clip(yf, 0, 0.62), bins=bins, orientation="horizontal", color="tab:blue", alpha=0.7)
    axy.set_xscale("log")
    axy.tick_params(labelleft=False)
    for t in THRESH.values():
        axx.axvline(t, color="0.6", lw=0.7, ls="--")
        axy.axhline(t, color="0.6", lw=0.7, ls="--")
    rho = spearmanr(xf, yf).statistic
    axx.set_title(f"{ylab}  vs  {xlab}\nSpearman ρ = {rho:.2f}  (clipped at 0.62; "
                  f"{int((xf > 0.62).sum())}/{int((yf > 0.62).sum())} x/y beyond)", fontsize=9)
    return rho


def main():
    an = si.load_sorting_analyzer(str(ANALYZER))
    qm = an.get_extension("quality_metrics").get_data()
    print("columns:", list(qm.columns), flush=True)
    isi = qm["isi_violations_ratio"].to_numpy()
    rp = qm["rp_contamination"].to_numpy()
    srp = qm["sliding_rp_violation"].to_numpy()

    def fin(*a):
        return np.logical_and.reduce([np.isfinite(v) for v in a])

    # ---- quantify the AND/OR disagreement per tier (isi vs rp) ----
    print("\n=== isi_violations_ratio vs rp_contamination: AND/OR per tier ===", flush=True)
    print(f"{'tier':<13}{'thr':>5}{'AND':>6}{'OR':>6}{'rp_ok_isi_bad':>15}{'isi_ok_rp_bad':>15}", flush=True)
    f2 = fin(isi, rp)
    for t, thr in THRESH.items():
        isi_ok = (isi <= thr) & f2
        rp_ok = (rp <= thr) & f2
        AND = isi_ok & rp_ok
        OR = (isi_ok | rp_ok) & f2
        print(f"{t:<13}{thr:>5}{int(AND.sum()):>6}{int(OR.sum()):>6}"
              f"{int((rp_ok & ~isi_ok).sum()):>15}{int((isi_ok & ~rp_ok).sum()):>15}", flush=True)

    print("\n=== sliding_rp_violation vs rp_contamination: AND/OR per tier ===", flush=True)
    f3 = fin(srp, rp)
    for t, thr in THRESH.items():
        s_ok = (srp <= thr) & f3
        r_ok = (rp <= thr) & f3
        print(f"{t:<13}{thr:>5}  AND={int((s_ok & r_ok).sum())}  OR={int((s_ok | r_ok).sum())}  "
              f"srp_ok_rp_bad={int((s_ok & ~r_ok).sum())}  rp_ok_srp_bad={int((r_ok & ~s_ok).sum())}", flush=True)

    # ---- figure ----
    fig = plt.figure(figsize=(15, 6.5))
    outer = fig.add_gridspec(1, 2, wspace=0.22)
    joint_panel(fig, outer[0, 0], rp, srp, "rp_contamination", "sliding_rp_violation", f3)
    joint_panel(fig, outer[0, 1], isi, rp, "isi_violations_ratio", "rp_contamination", f2)
    fig.suptitle("Refractory-contamination metric relationships (dashed = tier thresholds 0.1/0.3/0.5)",
                 fontsize=12)
    out = T / "refractory_metric_relationships.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

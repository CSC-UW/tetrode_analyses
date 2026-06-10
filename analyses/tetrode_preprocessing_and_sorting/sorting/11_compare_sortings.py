"""Compare the two compression stores' sortings (lossless blosc vs lossy wavpack).

Both sortings are of the SAME underlying recording, so a high agreement means
WavPack(bps=2.25) lossy compression did not materially change the sorting.
Uses spikeinterface.comparison.compare_two_sorters.
"""
import json
import pathlib
import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SORT_ROOT = ROOT / "sortings"
OUTDIR = pathlib.Path(__file__).resolve().parent

s_blosc = si.load(str(SORT_ROOT / "blosc-zstd" / "aggregated"))
s_wp = si.load(str(SORT_ROOT / "wavpack-bps2.25" / "aggregated"))
print(f"blosc(lossless): {s_blosc.get_num_units()} units | wavpack(lossy): {s_wp.get_num_units()} units")

cmp = sc.compare_two_sorters(
    s_blosc, s_wp, sorting1_name="blosc-lossless", sorting2_name="wavpack-lossy",
    delta_time=0.4, match_score=0.5,
)
matched = cmp.get_matching()[0]  # hungarian match for sorting1 -> sorting2 (-1 = unmatched)
n_matched = int((matched.values != -1).sum())
agreement = cmp.agreement_scores
summary = {
    "n_units_blosc": int(s_blosc.get_num_units()),
    "n_units_wavpack": int(s_wp.get_num_units()),
    "n_matched_units": n_matched,
    "n_unmatched_blosc": int(s_blosc.get_num_units() - n_matched),
    "n_unmatched_wavpack": int(s_wp.get_num_units() - n_matched),
    "mean_agreement_of_matched": float(
        np.nanmean([agreement.loc[u1, u2] for u1, u2 in matched.items() if u2 != -1])
    ) if n_matched else float("nan"),
}
print(json.dumps(summary, indent=2))
(SORT_ROOT / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

# agreement matrix figure: ordered heatmap so best-matched pairs lie on the diagonal.
# Hundreds of tetrode-aggregated units, so we drop per-unit tick labels (unreadable) and
# per-cell score text; a fixed 0-1 viridis scale + colorbar makes the diagonal legible.
scores = cmp.get_ordered_agreement_scores()  # rows/cols sorted by best agreement


def _sparse_ticks(n, k=6):
    """~k ordinal-position ticks for scale reference (not unit IDs)."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(
    scores.values, cmap="viridis", vmin=0.0, vmax=1.0,
    aspect="auto", interpolation="nearest",
)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("agreement score (Jaccard)")

xt = _sparse_ticks(scores.shape[1])
yt = _sparse_ticks(scores.shape[0])
ax.set_xticks(xt)
ax.set_xticklabels(xt)
ax.set_yticks(yt)
ax.set_yticklabels(yt)

ax.set_xlabel(f"wavpack-lossy unit (n={scores.shape[1]})")
ax.set_ylabel(f"blosc-lossless unit (n={scores.shape[0]})")
ax.set_title(
    "Agreement: blosc (lossless) vs wavpack (bps=2.25, lossy)\n"
    f"{n_matched} matched units, "
    f"mean matched agreement = {summary['mean_agreement_of_matched']:.3f}"
)
fig.tight_layout()
fig.savefig(OUTDIR / "sorting_agreement_matrix.png", dpi=150)
print("saved", OUTDIR / "sorting_agreement_matrix.png")
print("DONE")

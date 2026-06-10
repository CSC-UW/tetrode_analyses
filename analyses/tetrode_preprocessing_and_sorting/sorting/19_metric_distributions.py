"""Quality-metric distributions on the lossless reference sort (blosc-A, 781 units).

Computes SpikeInterface quality metrics (the set in wisc_ecephys_tools postpro
configs) on the deterministic lossless sort, so curation thresholds for the
"well-isolated" subset can be chosen against THIS data's distributions before
locking them. Overlays the lab's permissive/moderate/conservative tiers
(ecephys/wne/siutils.py) plus a proposed SNR floor.

Recording = bandpass(300-6000) + global CMR of the blosc store (identical to what
the sorter saw, minus ms5's internal per-group whitening), materialized once to a
float32 binary for fast waveform/amplitude/PCA reads, then deleted.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spikeinterface as si
from tetrode_analyses.sorting import preprocess_for_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
REF_AGG = ROOT / "sortings_seed42_pcafix" / "blosc-A" / "aggregated"
OUTDIR = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "sortings_seed42_pcafix" / "_metrics_cmr_cache"
DISK_FLOOR_GB = 1800

# (metric, direction, (permissive, moderate, conservative)); direction: 'hi' = keep >=, 'lo' = keep <=
TIERS = {
    "snr":                  ("hi", (3.0, 4.0, 5.0)),    # proposed addition
    "isi_violations_ratio": ("lo", (0.5, 0.3, 0.1)),
    "rp_contamination":     ("lo", (0.5, 0.3, 0.1)),
    "nn_isolation":         ("hi", (0.7, 0.8, 0.9)),
    "amplitude_cutoff":     ("lo", (0.5, 0.5, 0.3)),
    "presence_ratio":       ("hi", (0.8, 0.9, 0.9)),
    "firing_rate":          ("hi", (0.2, 0.5, 0.5)),
    "num_spikes":           ("hi", (None, None, None)),  # context only, no tier cut
}
LOG_X = {"num_spikes", "firing_rate", "isi_violations_ratio", "amplitude_cutoff"}


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


fg = free_gb()
print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB", flush=True)
if fg < DISK_FLOOR_GB:
    raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")

# ---- materialize bandpass + global CMR once ----
print(f"[{time.strftime('%T')}] materializing bandpass+global-CMR -> {CACHE} ...", flush=True)
rec = si.read_zarr(str(BLOSC))
if CACHE.exists():
    shutil.rmtree(CACHE)
pp = preprocess_for_sorting(rec, cmr="global")
pp_mat = pp.save(format="binary", folder=str(CACHE), dtype="float32", n_jobs=96, progress_bar=True, overwrite=True)
sorting = si.load(str(REF_AGG))
print(f"[{time.strftime('%T')}] rec {pp_mat.get_num_channels()}ch {pp_mat.get_num_frames()/30000/3600:.1f}h | "
      f"sorting {sorting.get_num_units()} units", flush=True)

# ---- SortingAnalyzer + extensions ----
t0 = time.perf_counter()
an = si.create_sorting_analyzer(sorting, pp_mat, format="memory", sparse=True, method="radius", radius_um=100)
an.compute("random_spikes", max_spikes_per_unit=500, seed=0)
an.compute("noise_levels")
an.compute("waveforms", ms_before=1.0, ms_after=1.5)
an.compute("templates")
an.compute("spike_amplitudes")
print(f"[{time.strftime('%T')}] extensions done ({(time.perf_counter()-t0)/60:.1f} min); computing metrics...", flush=True)

# Fast metric set (non-PCA). The PCA-based isolation metric (nn_advanced) is
# flagged slow in SI and deferred to a follow-up if these don't separate cleanly.
METRICS = ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation",
           "rp_violation", "amplitude_cutoff"]
qm = an.compute("quality_metrics", metric_names=METRICS,
                metric_params={"presence_ratio": {"bin_duration_s": 180}})
df = qm.get_data()
df.to_csv(OUTDIR / "metric_distributions_blosc-A.csv")
print(f"[{time.strftime('%T')}] metrics columns: {list(df.columns)}", flush=True)

shutil.rmtree(CACHE, ignore_errors=True)
print(f"[{time.strftime('%T')}] removed materialize cache", flush=True)

# ---- distributions + tier pass-fractions ----
N = len(df)
summary = {"n_units": int(N), "metrics": {}}
present = [m for m in TIERS if m in df.columns]
ncol = 3
nrow = int(np.ceil(len(present) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
axes = np.atleast_1d(axes).ravel()
tier_names = ["permissive", "moderate", "conservative"]
colors = ["#2ca02c", "#ff7f0e", "#d62728"]

for ax, m in zip(axes, present):
    x = df[m].to_numpy(dtype="float64")
    finite = x[np.isfinite(x)]
    direction, thr = TIERS[m]
    q = {p: float(np.nanpercentile(finite, p)) for p in (5, 25, 50, 75, 95)} if finite.size else {}
    use_log = m in LOG_X and finite.size and (finite > 0).all()
    if use_log:
        bins = np.logspace(np.log10(max(finite.min(), 1e-3)), np.log10(finite.max() + 1e-9), 40)
        ax.set_xscale("log")
    else:
        hi = np.nanpercentile(finite, 99) if finite.size else 1.0
        bins = np.linspace(finite.min() if finite.size else 0, hi, 40)
    ax.hist(np.clip(finite, bins[0], bins[-1]), bins=bins, color="#888", alpha=0.8)
    ax.set_title(m)
    ax.set_ylabel("units")
    mp = {}
    for name, c, t in zip(tier_names, colors, thr):
        if t is None:
            continue
        ax.axvline(t, color=c, ls="--", lw=1.4, label=f"{name} {'≥' if direction=='hi' else '<'}{t}")
        passing = (finite >= t).sum() if direction == "hi" else (finite < t).sum()
        mp[name] = int(passing)
    ax.legend(fontsize=7)
    nan_n = int((~np.isfinite(x)).sum())
    summary["metrics"][m] = {"direction": direction, "percentiles": {str(k): round(v, 4) for k, v in q.items()},
                             "n_nan": nan_n, "pass_per_tier": mp}

for ax in axes[len(present):]:
    ax.axis("off")
fig.suptitle(f"blosc-A quality metrics (n={N} units) — tier thresholds overlaid", y=1.0)
fig.tight_layout()
fig.savefig(OUTDIR / "metric_distributions_blosc-A.png", dpi=130, bbox_inches="tight")
print(f"[{time.strftime('%T')}] saved figure + csv", flush=True)

# ---- combined good-unit counts per tier (all gated metrics AND'd) ----
gated = [m for m in present if m != "num_spikes" and TIERS[m][1][0] is not None]
combined = {}
for i, name in enumerate(tier_names):
    mask = np.ones(N, dtype=bool)
    for m in gated:
        x = df[m].to_numpy(dtype="float64")
        direction, thr = TIERS[m]
        t = thr[i]
        ok = (x >= t) if direction == "hi" else (x < t)
        ok = ok & np.isfinite(x)   # NaN fails the gate
        mask &= ok
    combined[name] = int(mask.sum())
summary["combined_good_units_per_tier"] = combined
summary["gated_metrics"] = gated
print("\nCOMBINED good units (all metrics AND'd): " + json.dumps(combined), flush=True)
(OUTDIR / "metric_distributions_summary.json").write_text(json.dumps(summary, indent=2))
print("ALL DONE", flush=True)

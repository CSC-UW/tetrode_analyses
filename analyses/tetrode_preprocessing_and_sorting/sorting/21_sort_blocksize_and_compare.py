"""Effect of scheme-3 block duration: lossless blosc re-sorted at 1800 s and 900 s
blocks, compared to the 3600 s reference (blosc-A).

Everything except scheme3_block_duration_sec is held fixed (lossless blosc, global
CMR, whitening_seed=42, deterministic PCA, float32 materialize). The determinism
ceiling is 1.0 (blosc-A vs blosc-B), so any disagreement here is purely the
block-duration effect.

The bandpass+global-CMR materialize is identical regardless of block size (block
duration only affects ms5's sorting stage), so the blosc cache is materialized once
and reused for both sorts (shared cmr_cache_dir).

Comparison vs blosc-A (3600 s), both raw (all units) and curated on the
well-isolated set defined on blosc-A (ISI + RP + firing-rate tiers; see 19_/20_).
"""
import json
import shutil
import time
import pathlib
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
HERE = pathlib.Path(__file__).resolve().parent
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
REF_AGG = SR / "blosc-A" / "aggregated"     # 3600 s reference
SHARED_CACHE = SR / "_blocksize_cmr_cache"
METRICS_CSV = HERE / "metric_distributions_blosc-A.csv"
DISK_FLOOR_GB = 1800
SEED = 42

BASE = dict(scheme="3", cmr="global", whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)
BLOCKS = [("blosc-1800s", 1800), ("blosc-900s", 900)]

TIER_THRESH = {  # (isi_violations_ratio <, rp_contamination <, firing_rate >=)
    "permissive": (0.5, 0.5, 0.2), "moderate": (0.3, 0.3, 0.5), "conservative": (0.1, 0.1, 0.5),
}


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


def disk_guard(label):
    fg = free_gb()
    print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB}) before {label}", flush=True)
    if fg < DISK_FLOOR_GB:
        print(f"ABORT: insufficient disk ({fg:.0f} GB) before {label}.", flush=True)
        raise SystemExit(2)


def run_sort(out_name, block_s):
    out = SR / out_name
    print(f"\n=== [{time.strftime('%T')}] sorting {out_name} (block={block_s}s, seed={SEED}) ===", flush=True)
    t0 = time.perf_counter()
    agg = sort_store(BLOSC, out, scheme3_block_duration_sec=block_s, cmr_cache_dir=SHARED_CACHE, **BASE)
    secs = time.perf_counter() - t0
    agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
    agg.save(folder=str(out / "aggregated"), overwrite=True)
    groups = np.asarray(agg.get_property("group"))
    info = {"minutes": round(secs / 60, 1), "block_s": block_s, "total_units": int(agg.get_num_units()),
            "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
    print(f"[{time.strftime('%T')}] {out_name}: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
    print("RESULT " + json.dumps({out_name: info}), flush=True)
    return agg, info


# good-unit sets defined on blosc-A (3600 s reference)
df = pd.read_csv(METRICS_CSV, index_col=0)
def good_ids(tier):
    isi_t, rp_t, fr_t = TIER_THRESH[tier]
    ok = (df["isi_violations_ratio"] < isi_t) & (df["rp_contamination"] < rp_t) & (df["firing_rate"] >= fr_t)
    ok &= np.isfinite(df["isi_violations_ratio"]) & np.isfinite(df["rp_contamination"]) & np.isfinite(df["firing_rate"])
    return df.index[ok].to_numpy()
GOOD = {t: good_ids(t) for t in TIER_THRESH}


def compare_vs_ref(ref, other, oname):
    cmp = sc.compare_two_sorters(ref, other, sorting1_name="ref3600", sorting2_name=oname, match_score=0.5)
    m = cmp.get_matching()[0]
    ag = cmp.agreement_scores
    ref_ids = np.asarray(ref.unit_ids)
    res = {}
    for label, ids in [("all_units", ref_ids), *[(t, GOOD[t]) for t in TIER_THRESH]]:
        ids = np.asarray([i for i in ids if i in set(ref_ids.tolist())])
        matched = [u for u in ids if m.get(u, -1) != -1]
        frac = len(matched) / len(ids) if len(ids) else float("nan")
        mean_ag = float(np.nanmean([ag.loc[u, m[u]] for u in matched])) if matched else float("nan")
        res[label] = {"n_ref": int(len(ids)), "matched": len(matched),
                      "match_frac": round(frac, 3), "mean_agreement": round(mean_ag, 4)}
    return res


# ---- materialize once, sort both block sizes ----
disk_guard("blosc materialize")
summary, aggs = {}, {}
for name, block_s in BLOCKS:
    aggs[name], summary[name] = run_sort(name, block_s)
    (SR / "sorting_blocksize_summary.json").write_text(json.dumps(summary, indent=2))
shutil.rmtree(SHARED_CACHE, ignore_errors=True)
print(f"[{time.strftime('%T')}] removed shared blosc cmr cache", flush=True)

# ---- compare each vs 3600 s reference (blosc-A) ----
ref = si.load(str(REF_AGG))
out = {"reference": "blosc-A (3600 s)", "good_unit_counts": {t: int(len(v)) for t, v in GOOD.items()},
       "ref_total_units": int(ref.get_num_units()), "results": {}}
for name, _ in BLOCKS:
    out["results"][name] = {"total_units": summary[name]["total_units"],
                            "vs_3600s": compare_vs_ref(ref, aggs[name], name)}
(SR / "comparison_blocksize_summary.json").write_text(json.dumps(out, indent=2))

print("\n--- BLOCK-SIZE AGREEMENT vs 3600 s (mean agreement / match fraction) ---", flush=True)
print(f"{'set':<13}{'n_ref':>7}" + "".join(f"{n:>20}" for n, _ in BLOCKS), flush=True)
for label in ["all_units", *TIER_THRESH]:
    nref = out["results"][BLOCKS[0][0]]["vs_3600s"][label]["n_ref"]
    cells = [f"{out['results'][n]['vs_3600s'][label]['mean_agreement']:.3f} / "
             f"{out['results'][n]['vs_3600s'][label]['match_frac']:.2f}" for n, _ in BLOCKS]
    print(f"{label:<13}{nref:>7}" + "".join(f"{c:>20}" for c in cells), flush=True)
print(f"\nunit counts: 3600s={ref.get_num_units()} | " +
      " | ".join(f"{n}={summary[n]['total_units']}" for n, _ in BLOCKS), flush=True)
print("ALL DONE", flush=True)

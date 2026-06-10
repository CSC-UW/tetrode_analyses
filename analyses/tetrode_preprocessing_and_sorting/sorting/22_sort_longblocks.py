"""Lossless blosc sorted at long scheme-3 block durations: 7200 s (2h), 14400 s
(4h), 21600 s (6h). Reports pure sort time + unit count for each.

The bandpass+global-CMR binary is materialized ONCE up front (timed separately),
then each sort reuses it via the shared cmr_cache_dir, so each reported sort time
is pure-sort (no materialize confound). Same fixed pipeline as the other runs:
global CMR, whitening_seed=42, deterministic PCA, float32 materialize.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import spikeinterface as si
from tetrode_analyses.sorting import preprocess_for_sorting, sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SHARED_CACHE = SR / "_longblocks_cmr_cache"
DISK_FLOOR_GB = 1800
SEED = 42
BASE = dict(scheme="3", cmr="global", whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)
BLOCKS = [("blosc-7200s", 7200), ("blosc-14400s", 14400), ("blosc-21600s", 21600)]


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


fg = free_gb()
print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB})", flush=True)
if fg < DISK_FLOOR_GB:
    raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")

# ---- materialize bandpass + global CMR ONCE (timed separately) ----
if SHARED_CACHE.exists():
    shutil.rmtree(SHARED_CACHE)
print(f"[{time.strftime('%T')}] materializing bandpass+global-CMR -> {SHARED_CACHE} ...", flush=True)
t0 = time.perf_counter()
rec = si.read_zarr(str(BLOSC))
pp = preprocess_for_sorting(rec, cmr="global")
pp.save(format="binary", folder=str(SHARED_CACHE), dtype="float32", n_jobs=96, progress_bar=True, overwrite=True)
mat_min = (time.perf_counter() - t0) / 60
print(f"[{time.strftime('%T')}] materialize done in {mat_min:.1f} min", flush=True)

# ---- sort each block size (pure sort, reusing cache) ----
summary = {"materialize_min": round(mat_min, 1), "sorts": {}}
for name, block_s in BLOCKS:
    out = SR / name
    print(f"\n=== [{time.strftime('%T')}] sorting {name} (block={block_s}s) ===", flush=True)
    t0 = time.perf_counter()
    agg = sort_store(BLOSC, out, scheme3_block_duration_sec=block_s, cmr_cache_dir=SHARED_CACHE, **BASE)
    sort_min = (time.perf_counter() - t0) / 60
    agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
    agg.save(folder=str(out / "aggregated"), overwrite=True)
    groups = np.asarray(agg.get_property("group"))
    info = {"block_s": block_s, "sort_min": round(sort_min, 1), "total_units": int(agg.get_num_units()),
            "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
    summary["sorts"][name] = info
    print(f"[{time.strftime('%T')}] {name}: {agg.get_num_units()} units in {sort_min:.1f} min (pure sort)", flush=True)
    print("RESULT " + json.dumps({name: {k: info[k] for k in ('block_s', 'sort_min', 'total_units')}}), flush=True)
    (SR / "sorting_longblocks_summary.json").write_text(json.dumps(summary, indent=2))

shutil.rmtree(SHARED_CACHE, ignore_errors=True)
print(f"[{time.strftime('%T')}] removed shared cache", flush=True)

print("\n--- LONG-BLOCK SORT (pure sort time / units) ---", flush=True)
print(f"materialize (one-time): {mat_min:.1f} min", flush=True)
print(f"{'block':<14}{'sort_min':>10}{'units':>8}", flush=True)
for name, block_s in BLOCKS:
    i = summary["sorts"][name]
    print(f"{name:<14}{i['sort_min']:>10.1f}{i['total_units']:>8}", flush=True)
print("ALL DONE", flush=True)

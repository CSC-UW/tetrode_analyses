"""Full 48 h lossless (blosc) sort with KILOSORT4, BY TETRODE GROUP (one KS4 run per
tetrode, like MS5), NO drift correction. Reports wall time and compute footprint (peak
USS/RSS/CPU + peak /nvme + peak GPU memory/utilization), split into the one-time
materialize phase and the sort phase. The Kilosort4 counterpart to
27_sort_48h_singleblock_scheme2.py (MountainSort5 scheme 2, single 48 h block).

Why by group (not together)? Sorting all 16 tetrodes in one 64-channel run is far
faster on the GPU, BUT KS4's final clustering then holds ALL 48 h x 64-ch spike
features on the GPU at once (~29 GiB) and needs a further ~12 GiB clustering
allocation -> ~42 GiB, which OOMs the 32 GB V100 (a hard wall; not fragmentation --
`expandable_segments` confirmed reserved-unallocated was ~46 MiB). The per-tetrode
clustering allocation (~12 GiB) is per spatial center and the SAME regardless of how
many tetrodes share a run; only the feature baseline scales with tetrode count. So
ONE tetrode per run (~1.8 GiB baseline + ~12 GiB clustering ~ 14 GiB) fits
comfortably. By-group is slower (16 detection passes, ~8-12 h) but robust and is the
exact per-tetrode analog of the MS5 reference.

Preprocessing is IDENTICAL to the MS5 runs: bandpass(300-6000 Hz) + global CMR,
float32, materialized ONCE to a shared binary cache (`materialize_preprocessed`); the
per-tetrode sorts read their 4 channels from it (rewritten to a contiguous 4-ch .dat
per tetrode for KS4's fast reader; the rewrite is parallelized via `write_n_jobs`).
KS4 differences (see `sort_store_ks4`): `do_CAR=False` (global CMR already applied,
so no second CAR), `do_correction=False`/`nblocks=0` (NO drift correction), KS4's
internal whitening kept (analog of MS5's per-group whiten), `batch_size=300_000`
(10 s, validated on a 10-min smoke crop), `nearest_chans=whitening_range=4`, `dminx=16`,
and `templates_from_data=False` (REQUIRED: the data-derived universal-template step
collects uncapped clips across the whole recording and its KMeans hangs for hours at
48 h; prefab templates seed detection instead, KS4 still learns real templates while
clustering). `run_sorter_by_property` tags each unit with its tetrode `group`.

GPU: runs on the host Tesla V100 (torch cu128, sm_70). The footprint sampler polls
`nvidia-smi` for GPU memory/util.

Footprint comparison target: MS5 scheme-2 single-block
(`sorting_singleblock_scheme2_summary.json`): materialize 32.7 min / sort 99.8 min
(CPU-bound n_jobs=5, no GPU), peak USS ~82 GB, peak /nvme ~2.0 TB.
"""
import os
# clear_cache=True frees torch's reserved GPU memory between KS4's memory-intensive
# ops; expandable_segments avoids allocator fragmentation. Neither is load-bearing
# for the by-group path (one tetrode fits easily) but both are harmless safety
# margin. Must be set BEFORE torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import shutil
import subprocess
import threading
import time
import pathlib
import numpy as np
import psutil
import torch
import kilosort
from tetrode_analyses.sorting import materialize_preprocessed, sort_store_ks4

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SHARED_CACHE = SR / "_ks4_cmr_cache"
OUT = SR / "blosc-ks4-nodrift"
DISK_FLOOR_GB = 1800
GPU_ID = 0
# Sort BY TETRODE GROUP: one KS4 run per tetrode (4 ch), like MS5. Each tetrode's
# clustering fits the 32 GB GPU; sorting together OOMs (see module docstring).
# templates_from_data=False is REQUIRED (the data-derived universal-template KMeans
# hangs for hours at 48 h); clear_cache frees reserved GPU memory between ops.
KS4 = dict(
    batch_size=300_000, do_CAR=False, do_correction=False, nblocks=0,
    skip_kilosort_preprocessing=False, nearest_chans=4, whitening_range=4,
    dminx=16.0, torch_device="auto",
    grouping_property="group",  # one KS4 run per tetrode (by-group, like MS5)
    use_binary_file=True,       # rewrite a contiguous 4-ch .dat per tetrode (KS4 fast reader)
    sort_n_jobs=1,              # tetrodes sorted sequentially -> one KS4 process owns the GPU
    write_n_jobs=16,            # parallelize the per-tetrode .dat rewrite
    sorter_params={"templates_from_data": False, "clear_cache": True},
)


def _gpu_sample(gpu_id=GPU_ID):
    """(device memory.used MB, utilization.gpu %) for one GPU; (0,0) on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        mem_s, util_s = out.stdout.strip().split(",")
        return float(mem_s), float(util_s)
    except Exception:
        return 0.0, 0.0


class TreeSampler:
    """Peak USS (private working set; excludes mmap'd file cache), RSS, CPU over the
    process tree, peak /nvme usage (min free seen), and peak GPU device memory.used /
    utilization (whole-GPU via nvidia-smi -- accurate as the KS footprint because the
    sort runs one KS process at a time on an otherwise idle V100)."""
    def __init__(self, interval=1.0, disk_path="/nvme"):
        self.interval = interval
        self.disk_path = disk_path
        self.proc = psutil.Process()
        self._run = False
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
        self.peak_gpu_mem_mb = 0.0
        self.peak_gpu_util = 0.0
        self.min_free_disk = shutil.disk_usage(disk_path).free

    def _loop(self):
        while self._run:
            try:
                procs = [self.proc] + self.proc.children(recursive=True)
            except psutil.Error:
                procs = [self.proc]
            uss = rss = 0
            cpu = 0.0
            for p in procs:
                try:
                    mi = p.memory_full_info()
                    uss += mi.uss
                    rss += mi.rss
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    pass
            self.peak_uss = max(self.peak_uss, uss)
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_cpu = max(self.peak_cpu, cpu)
            gmem, gutil = _gpu_sample()
            self.peak_gpu_mem_mb = max(self.peak_gpu_mem_mb, gmem)
            self.peak_gpu_util = max(self.peak_gpu_util, gutil)
            self.min_free_disk = min(self.min_free_disk, shutil.disk_usage(self.disk_path).free)
            time.sleep(self.interval)

    def start(self):
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
        self.peak_gpu_mem_mb = 0.0
        self.peak_gpu_util = 0.0
        self.min_free_disk = shutil.disk_usage(self.disk_path).free
        for p in [self.proc] + self.proc.children(recursive=True):
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        self._t.join(timeout=5)

    def snapshot(self, start_free):
        return {"peak_uss_gb": round(self.peak_uss / 1e9, 1),
                "peak_rss_gb_incl_filecache": round(self.peak_rss / 1e9, 1),
                "peak_cpu_cores": round(self.peak_cpu / 100, 1),
                "peak_disk_used_gb": round((start_free - self.min_free_disk) / 1e9, 1),
                "peak_gpu_mem_mb": round(self.peak_gpu_mem_mb, 1),
                "peak_gpu_util_pct": round(self.peak_gpu_util, 1)}


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


def max_spike_sample_index(sorting):
    """Largest spike sample index over all units -- a 48 h (5.215e9-sample) int
    overflow tripwire. KS4 stores spike times as float64 sec + int64 batch offset,
    so this should be ~5.2e9, NOT wrapped to a small value."""
    mx = 0
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u)
        if st.size:
            mx = max(mx, int(st.max()))
    return mx


def main():
    fg = free_gb()
    print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB}) | "
          f"RAM total {psutil.virtual_memory().total/1e9:.0f} GB | "
          f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | ks {kilosort.__version__}",
          flush=True)
    if fg < DISK_FLOOR_GB:
        raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")

    summary = {"sorter": "kilosort4", "ks_version": kilosort.__version__,
               "torch_version": torch.__version__,
               "gpu": torch.cuda.get_device_name(0),
               "layout": "by-group (one KS4 run per tetrode, like MS5)", "ks_params": dict(KS4),
               "drift_correction": False,
               "ram_total_gb": round(psutil.virtual_memory().total / 1e9),
               "metric": "peak USS (private working set, summed over process tree); GPU is device memory.used"}

    # ---- Phase 1: materialize bandpass + global CMR ONCE (timed + sampled), or
    # resume an existing cache. bandpass + global CMR is deterministic, so a valid
    # cache (e.g. from a prior run / iteration) is reused as-is. ----
    fresh = not SHARED_CACHE.exists()
    print(f"[{time.strftime('%T')}] {'materializing' if fresh else 'reusing'} bandpass+global-CMR "
          f"@ {SHARED_CACHE} ...", flush=True)
    sampler = TreeSampler(interval=1.0)
    start_free = shutil.disk_usage("/nvme").free
    sampler.start()
    t0 = time.perf_counter()
    materialize_preprocessed(BLOSC, SHARED_CACHE, cmr="global", materialize_n_jobs=96)
    mat_min = (time.perf_counter() - t0) / 60
    sampler.stop()
    if fresh:
        summary["materialize_min"] = round(mat_min, 1)
        summary["materialize_footprint"] = sampler.snapshot(start_free)
        print(f"[{time.strftime('%T')}] materialize done in {mat_min:.1f} min | {summary['materialize_footprint']}", flush=True)
    else:
        summary["materialize_min"] = "reused"
        summary["materialize_footprint"] = "reused (deterministic; measured fresh at 34.9 min / USS 84.3 GB / 1379 GB /nvme)"
        print(f"[{time.strftime('%T')}] reused materialize cache (load {mat_min:.2f} min)", flush=True)

    # ---- Phase 2: KS4 sort full 48 h by tetrode group, no drift (sampled) ----
    print(f"\n=== [{time.strftime('%T')}] KS4 sorting full 48 h BY GROUP (16 tetrodes), no drift, batch_size={KS4['batch_size']} ===", flush=True)
    sampler = TreeSampler(interval=1.0)
    start_free = shutil.disk_usage("/nvme").free
    sampler.start()
    t0 = time.perf_counter()
    # run_sorter_by_property tags each unit with its tetrode `group` -> no post-hoc
    # assignment needed (unlike the together path).
    agg = sort_store_ks4(BLOSC, OUT, cmr="global", cmr_cache_dir=SHARED_CACHE,
                         materialize_n_jobs=96, **KS4)
    sort_min = (time.perf_counter() - t0) / 60
    sampler.stop()

    agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
    agg.save(folder=str(OUT / "aggregated"), overwrite=True)
    groups = np.asarray(agg.get_property("group"))
    summary["sort_min"] = round(sort_min, 1)
    summary["sort_footprint"] = sampler.snapshot(start_free)
    summary["total_units"] = int(agg.get_num_units())
    summary["per_tetrode"] = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
    summary["max_spike_sample_index"] = max_spike_sample_index(agg)
    print(f"[{time.strftime('%T')}] sort done: {agg.get_num_units()} units in {sort_min:.1f} min | "
          f"{summary['sort_footprint']}", flush=True)
    print(f"[{time.strftime('%T')}] max spike sample index = {summary['max_spike_sample_index']} "
          f"(expect ~5.215e9; tripwire for int overflow)", flush=True)

    (SR / "sorting_ks4_summary.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(SHARED_CACHE, ignore_errors=True)
    print(f"[{time.strftime('%T')}] removed shared cache", flush=True)

    print("\n--- 48 h KS4 SORT (by group, no drift) ---", flush=True)
    print(f"materialize : {mat_min:.1f} min | {summary['materialize_footprint']}", flush=True)
    print(f"sort        : {sort_min:.1f} min | {summary['sort_footprint']}", flush=True)
    print(f"total units : {summary['total_units']}", flush=True)
    print("RESULT " + json.dumps({k: summary[k] for k in
          ("materialize_min", "sort_min", "sort_footprint", "total_units", "max_spike_sample_index")}), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

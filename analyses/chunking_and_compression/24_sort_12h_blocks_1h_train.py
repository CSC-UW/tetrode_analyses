"""Full 48 h lossless (blosc) sort at scheme-3 block_duration = 12 h (43200 s)
with a 1 h (3600 s) uniform training window per block. Reports wall time and
compute footprint (peak USS/RSS/CPU + peak /nvme usage), split into the
one-time materialize phase and the sort phase.

Rationale (see SORTING_COMPARISON_FINDINGS.md / the methodology discussion):
block_duration_sec is the stationarity window; scheme2_training_duration_sec is
the per-block unit-discovery/template-training budget. 12 h blocks (43200 s) and
the 1 h training window are both well under the ms5 int32 detect_spikes ceiling
(~19.9 h = 71,583 s) -- and within a block, phase-2 classification is chunked
(~833 s/chunk for 4 ch), so no single detect_spikes call approaches 2^31 samples.

Same fixed production pipeline as the other seeded runs: global CMR,
whitening_seed=42, deterministic PCA, float32 materialize, sort_n_jobs=5.
"""
import json
import shutil
import threading
import time
import pathlib
import numpy as np
import psutil
import spikeinterface as si
from tetrode_analyses.sorting import preprocess_for_sorting, sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SHARED_CACHE = SR / "_12hblock_cmr_cache"
OUT = SR / "blosc-43200s-train3600s"
DISK_FLOOR_GB = 1800
SEED = 42
BLOCK_S = 43200      # 12 h
TRAIN_S = 3600       # 1 h
BASE = dict(scheme="3", cmr="global", whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)


class TreeSampler:
    """Peak USS (private working set; excludes mmap'd file cache), RSS, CPU over
    the process tree, plus peak /nvme usage (min free seen)."""
    def __init__(self, interval=1.0, disk_path="/nvme"):
        self.interval = interval
        self.disk_path = disk_path
        self.proc = psutil.Process()
        self._run = False
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
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
            self.min_free_disk = min(self.min_free_disk, shutil.disk_usage(self.disk_path).free)
            time.sleep(self.interval)

    def start(self):
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
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
                "peak_disk_used_gb": round((start_free - self.min_free_disk) / 1e9, 1)}


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


fg = free_gb()
print(f"[{time.strftime('%T')}] /nvme free {fg:.0f} GB (floor {DISK_FLOOR_GB}) | "
      f"RAM total {psutil.virtual_memory().total/1e9:.0f} GB", flush=True)
if fg < DISK_FLOOR_GB:
    raise SystemExit(f"ABORT: insufficient disk ({fg:.0f} GB)")

summary = {"block_s": BLOCK_S, "train_s": TRAIN_S, "seed": SEED, "sort_n_jobs": BASE["sort_n_jobs"],
           "ram_total_gb": round(psutil.virtual_memory().total / 1e9),
           "metric": "peak USS (private working set, summed over process tree)"}

# ---- Phase 1: materialize bandpass + global CMR ONCE (timed + sampled) ----
if SHARED_CACHE.exists():
    shutil.rmtree(SHARED_CACHE)
print(f"[{time.strftime('%T')}] materializing bandpass+global-CMR -> {SHARED_CACHE} ...", flush=True)
sampler = TreeSampler(interval=1.0)
start_free = shutil.disk_usage("/nvme").free
sampler.start()
t0 = time.perf_counter()
rec = si.read_zarr(str(BLOSC))
pp = preprocess_for_sorting(rec, cmr="global")
pp.save(format="binary", folder=str(SHARED_CACHE), dtype="float32", n_jobs=96, progress_bar=True, overwrite=True)
mat_min = (time.perf_counter() - t0) / 60
sampler.stop()
summary["materialize_min"] = round(mat_min, 1)
summary["materialize_footprint"] = sampler.snapshot(start_free)
print(f"[{time.strftime('%T')}] materialize done in {mat_min:.1f} min | {summary['materialize_footprint']}", flush=True)

# ---- Phase 2: sort full 48 h at 12 h blocks / 1 h training (pure sort, sampled) ----
print(f"\n=== [{time.strftime('%T')}] sorting full 48 h: block={BLOCK_S}s (12 h), train={TRAIN_S}s (1 h) ===", flush=True)
sampler = TreeSampler(interval=1.0)
start_free = shutil.disk_usage("/nvme").free
sampler.start()
t0 = time.perf_counter()
agg = sort_store(
    BLOSC, OUT,
    scheme3_block_duration_sec=BLOCK_S,
    cmr_cache_dir=SHARED_CACHE,
    sorter_params={"scheme2_training_duration_sec": TRAIN_S,
                   "scheme2_training_recording_sampling_mode": "uniform"},
    **BASE,
)
sort_min = (time.perf_counter() - t0) / 60
sampler.stop()

agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
agg.save(folder=str(OUT / "aggregated"), overwrite=True)
groups = np.asarray(agg.get_property("group"))
summary["sort_min"] = round(sort_min, 1)
summary["sort_footprint"] = sampler.snapshot(start_free)
summary["total_units"] = int(agg.get_num_units())
summary["per_tetrode"] = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
print(f"[{time.strftime('%T')}] sort done: {agg.get_num_units()} units in {sort_min:.1f} min | "
      f"{summary['sort_footprint']}", flush=True)

(SR / "sorting_12hblock_train1h_summary.json").write_text(json.dumps(summary, indent=2))
shutil.rmtree(SHARED_CACHE, ignore_errors=True)
print(f"[{time.strftime('%T')}] removed shared cache", flush=True)

print("\n--- 48 h SORT @ 12 h blocks / 1 h training ---", flush=True)
print(f"materialize : {mat_min:.1f} min | {summary['materialize_footprint']}", flush=True)
print(f"sort        : {sort_min:.1f} min | {summary['sort_footprint']}", flush=True)
print(f"total units : {summary['total_units']}", flush=True)
print("RESULT " + json.dumps({k: summary[k] for k in
      ("block_s", "train_s", "materialize_min", "sort_min", "sort_footprint", "total_units")}), flush=True)
print("ALL DONE", flush=True)

"""Footprint of scheme-2 training_duration_sec on tononi-2 (runtime + peak RSS/CPU).

Finds the largest training_duration the host can realistically support. Per the
ms5 scheme-2 logic (sorting_scheme2.py:72-120), training_duration_sec sets how much
data trains the per-block classifiers AND phase-1 clustering; the training traces
(training_duration x 30 kHz x 4 ch x f32) plus the detected-spike snippet set scale
~linearly with it, so the footprint should grow ~linearly with training_duration.

Method: for each training_duration T, materialize a *genuine T-second crop* of the
blosc bandpass+global-CMR binary (small, T-sized -- NOT a frame-slice of the full
48 h binary, which ms5 loads in full), then sort all 16 tetrodes with MountainSort5
scheme 2 (training on the whole crop -> isolates the training cost) at production
n_jobs=5. A psutil sampler over the process tree records peak total **USS** (private
working set; excludes the mmap'd file cache, the right "real RAM" metric), peak CPU,
and wall time. Sweep doubles from the 300 s default and stops at a courteous RAM
budget or a per-point runtime cap (tononi-2 is a shared 1.5 TiB host).

NOTE: tetrode-specific (4 ch/group). Denser per-shank probes would scale the
per-group footprint up substantially.
"""
import json
import shutil
import threading
import time
import pathlib
import psutil
import spikeinterface as si
import spikeinterface.sorters as ss
from tetrode_analyses.sorting import preprocess_for_sorting  # also applies the ms5 _segments patch

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SR = ROOT / "sortings_seed42_pcafix"
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
CACHE = SR / "_traindur_crop"
WORK = SR / "_traindur_work"
FS = 30000
N_FRAMES = 5215033052
SEED = 42
DISK_FLOOR_GB = 1500
BUDGET_GB = 500.0          # courteous peak-USS budget on the shared 1.5 TiB host
RUNTIME_CAP_MIN = 90.0     # stop sweep if a single point exceeds this
N_JOBS = 5                 # production tetrode parallelism
TS = [300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 76800]
TS = [t for t in TS if t * FS <= N_FRAMES]


class TreeSampler:
    # Primary metric is USS (unique/private set size) summed over the process tree:
    # it excludes memory-mapped file-cache pages (shared, reclaimable) and so reflects
    # the sort's real anonymous working set. RSS kept only for reference/contrast.
    def __init__(self, interval=1.0):
        self.interval = interval
        self.proc = psutil.Process()
        self._run = False
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0

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
            time.sleep(self.interval)

    def start(self):
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
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


def free_gb(p="/nvme"):
    return shutil.disk_usage(p).free / 1e9


print(f"[{time.strftime('%T')}] /nvme free {free_gb():.0f} GB | RAM total {psutil.virtual_memory().total/1e9:.0f} GB", flush=True)

SORT_KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=SEED,
               scheme2_training_recording_sampling_mode="uniform", detect_threshold=5.5, detect_sign=-1)

results = []
sampler = TreeSampler(interval=1.0)
for T in TS:
    if free_gb() < DISK_FLOOR_GB:
        print(f"STOP: insufficient disk before T={T}", flush=True)
        break
    # ---- materialize a genuine T-second crop (small binary) ----
    shutil.rmtree(CACHE, ignore_errors=True)
    crop = si.read_zarr(str(BLOSC)).frame_slice(0, int(T * FS))
    crop.reset_times()  # drop the (full-length) time vector so the crop save doesn't trip set_times
    print(f"\n=== [{time.strftime('%T')}] T={T}s ({T/3600:.2f} h): materializing crop ===", flush=True)
    tm = time.perf_counter()
    pp_T = preprocess_for_sorting(crop, cmr="global").save(
        format="binary", folder=str(CACHE), dtype="float32", n_jobs=96, progress_bar=False, overwrite=True)
    mat_min = (time.perf_counter() - tm) / 60

    wf = WORK / f"T{T}"
    shutil.rmtree(wf, ignore_errors=True)
    print(f"[{time.strftime('%T')}] crop frames={pp_T.get_num_frames()} | sorting 16 tetrodes n_jobs={N_JOBS}", flush=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        agg = ss.run_sorter_by_property("mountainsort5", pp_T, grouping_property="group", folder=str(wf),
                                        engine="joblib", engine_kwargs={"n_jobs": N_JOBS}, verbose=False,
                                        scheme2_training_duration_sec=T, **SORT_KW)
        n_units = int(agg.get_num_units())
    finally:
        sort_min = (time.perf_counter() - t0) / 60
        sampler.stop()
    peak_uss_gb = sampler.peak_uss / 1e9
    rec = {"training_duration_s": T, "crop_h": round(T / 3600, 2), "n_jobs": N_JOBS,
           "materialize_min": round(mat_min, 1), "sort_min": round(sort_min, 1),
           "peak_uss_gb": round(peak_uss_gb, 1), "peak_rss_gb_incl_filecache": round(sampler.peak_rss / 1e9, 1),
           "peak_cpu_cores": round(sampler.peak_cpu / 100, 1), "n_units": n_units,
           "uss_per_tetrode_gb": round(peak_uss_gb / N_JOBS, 2)}
    results.append(rec)
    shutil.rmtree(wf, ignore_errors=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    print("RESULT " + json.dumps(rec), flush=True)
    (SR / "training_duration_bench.json").write_text(json.dumps(
        {"ram_total_gb": round(psutil.virtual_memory().total / 1e9), "budget_gb": BUDGET_GB,
         "metric": "peak_uss_gb (private working set, summed over process tree)", "results": results}, indent=2))
    if peak_uss_gb > BUDGET_GB:
        print(f"STOP: peak USS {peak_uss_gb:.0f} GB exceeded budget {BUDGET_GB:.0f} GB", flush=True)
        break
    if sort_min > RUNTIME_CAP_MIN:
        print(f"STOP: sort {sort_min:.0f} min exceeded cap {RUNTIME_CAP_MIN:.0f} min", flush=True)
        break

shutil.rmtree(CACHE, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)

print("\n--- TRAINING_DURATION FOOTPRINT (tononi-2, n_jobs=5, 4-ch tetrodes) ---", flush=True)
print("memory = peak USS (private working set; excludes mmap'd file cache)", flush=True)
print(f"{'train_s':>8}{'crop_h':>8}{'sort_min':>9}{'peak_USS_GB':>12}{'USS/tt_GB':>11}{'CPU_cores':>11}{'units':>7}", flush=True)
for r in results:
    print(f"{r['training_duration_s']:>8}{r['crop_h']:>8.2f}{r['sort_min']:>9.1f}{r['peak_uss_gb']:>12.1f}"
          f"{r['uss_per_tetrode_gb']:>11.2f}{r['peak_cpu_cores']:>11.1f}{r['n_units']:>7}", flush=True)
print("ALL DONE", flush=True)

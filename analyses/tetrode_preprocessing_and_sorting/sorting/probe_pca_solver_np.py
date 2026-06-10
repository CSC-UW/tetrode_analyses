"""600 s-window PCA-solver probe on Neuropixels (CNPIX12-Santiago imec0),
faithful to the MountainSort5 NP quickstart
(github.com/flatironinstitute/mountainsort5 examples/neuropixel_quickstart/spikeglx.py).

Replicates the example's pipeline and parameters exactly: phase_shift ->
bandpass(500,12000) -> detect_bad_channels + remove -> whiten -> save to binary
-> ms5.sorting_scheme2 (direct API, not run_sorter) with the example's
Scheme2SortingParameters (max_num_snippets_per_training_batch=1000,
snippet_mask_radius=60, classifier_npca=10, detect_sign=0, snippet_T1=15/T2=40,
training mode 'uniform', training_duration_sec=350, ...).

Deviations (unavoidable / noted): (1) input is a Zarr SI recording (read_zarr),
not SpikeGLX; (2) cropped to CROP_S because materializing the full 2-day
preprocessed binary would be ~7.9 TB (> free disk). The 350 s 'uniform' training
sample still spreads across the crop window. ms5.sorting_scheme2 runs in-process,
so SnippetClassifier.fit() is instrumented via monkeypatch.
"""
import csv
import time
import shutil
import pathlib
import collections
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import threadpoolctl
import spikeinterface.full as si
import mountainsort5 as ms5
import mountainsort5.core.SnippetClassifier as scmod

NP = "/Volumes/npx_nfs/nobak/shared/novel_objects_deprivation/CNPIX12-Santiago/imec0.si_recording.zarr"
OUT = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/solver_probe_np")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "solver_np.csv"
if LOG.exists():
    LOG.unlink()
CROP_S = 1200.0  # full 2-day preprocessed binary would be ~7.9 TB; crop to fit disk

_orig_fit = scmod.SnippetClassifier.fit
def _logged_fit(self):
    r = _orig_fit(self)
    try:
        with open(LOG, "a") as f:
            f.write(f"{len(self.all_training_labels)},{self.T},{self.M},"
                    f"{int(self.pca_model.n_features_in_)},{int(self.pca_model.n_components_)},"
                    f"{len(self.training_batches)},{getattr(self.pca_model,'_fit_svd_solver','?')}\n")
    except Exception:
        pass
    return r
scmod.SnippetClassifier.fit = _logged_fit

fs = 30000.004590
rec = si.read_zarr(NP).frame_slice(0, int(CROP_S * fs))
print(f"[{time.strftime('%T')}] loaded {rec.get_num_channels()} ch x {rec.get_total_duration():.0f} s", flush=True)
try:
    rec = si.phase_shift(rec)
    print("    + phase_shift applied", flush=True)
except Exception as e:
    print(f"    ! phase_shift skipped ({type(e).__name__}: {e})", flush=True)
rec = si.bandpass_filter(rec, freq_min=500, freq_max=12000)
bad_channel_ids, channel_labels = si.detect_bad_channels(rec)
rec = rec.remove_channels(bad_channel_ids)
print(f"    + bad channels removed: {len(bad_channel_ids)} -> {rec.get_num_channels()} good", flush=True)
rec = si.whiten(rec, dtype="float32")

BIN = OUT / "rec_bin"
if BIN.exists():
    shutil.rmtree(BIN)
print(f"[{time.strftime('%T')}] saving preprocessed binary ...", flush=True)
# Fork-safe multithreading: limit BLAS to 1 thread ONLY around the fork-based save
# (I/O-bound, so no speed loss), then full BLAS threading is restored for the PCA.
# Avoids the OpenBLAS-after-fork deadlock without globally single-threading.
with threadpoolctl.threadpool_limits(limits=1):
    cached = rec.save(folder=str(BIN), format="binary", n_jobs=16, chunk_duration="1s", progress_bar=True)
print(f"[{time.strftime('%T')}] preprocessing complete; sorting (scheme 2)...", flush=True)

sorting_params = {
    "max_num_snippets_per_training_batch": 1000,
    "snippet_mask_radius": 60,
    "phase1_npca_per_channel": 3,
    "phase1_npca_per_subdivision": 10,
    "classifier_npca": 10,
    "detect_channel_radius": 60,
    "phase1_detect_channel_radius": 60,
    "training_recording_sampling_mode": "uniform",
    "training_duration_sec": 350,
    "phase1_detect_threshold": 5.5,
    "detect_threshold": 5.25,
    "snippet_T1": 15,
    "snippet_T2": 40,
    "detect_sign": 0,
    "phase1_detect_time_radius_msec": 0.5,
    "detect_time_radius_msec": 0.5,
    "classification_chunk_sec": 100,
}
t0 = time.perf_counter()
sorting = ms5.sorting_scheme2(recording=cached, sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params))
print(f"[{time.strftime('%T')}] sorted in {(time.perf_counter()-t0)/60:.1f} min, {sorting.get_num_units()} units", flush=True)

rows = []
with open(LOG) as f:
    for r in csv.reader(f):
        if len(r) == 7:
            rows.append(r)
from sklearn.decomposition import PCA
auto_cache = {}
def auto_solver(L, nfeat, ncomp):
    k = (L, nfeat, ncomp)
    if k not in auto_cache:
        p = PCA(n_components=ncomp, svd_solver="auto"); p.fit(np.zeros((L, nfeat), dtype=np.float32))
        auto_cache[k] = p._fit_svd_solver
    return auto_cache[k]
L = np.array([int(r[0]) for r in rows]); nfeat = np.array([int(r[3]) for r in rows]); ncomp = np.array([int(r[4]) for r in rows])
actual = collections.Counter(r[6] for r in rows)
would = collections.Counter(auto_solver(int(r[0]), int(r[3]), int(r[4])) for r in rows)
print(f"\n=== NP scheme2 probe (quickstart params, crop {CROP_S:.0f}s): {len(rows)} classifier fits ===")
print(f"  M (neighborhood ch) distinct: {sorted(set(int(r[2]) for r in rows))}")
print(f"  n_features (T*M) distinct: {sorted(set(nfeat.tolist()))}")
print(f"  n_components distinct: {sorted(set(ncomp.tolist()))}")
print(f"  L: min={L.min()} p25={int(np.percentile(L,25))} median={int(np.median(L))} p75={int(np.percentile(L,75))} max={L.max()}")
print(f"  ACTUAL solver (covariance_eigh fix active): {dict(actual)}")
print(f"  WOULD pick under sklearn 'auto' (no fix): {dict(would)}")
print(f"  >>> 'auto' randomized fraction: {would.get('randomized',0)/max(len(rows),1):.1%}")
print("PROBE NP DONE", flush=True)

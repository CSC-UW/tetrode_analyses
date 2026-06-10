"""Stage 6: confirm the per-worker time-vector multiplication via the REAL
run_sorter_by_property joblib path, and serve as the before/after harness for
the np.load(mmap_mode) fix.

Cheap scale: NCH=8 (2 groups), n_jobs=2, N=3e8 -> time vector = 2.4 GB,
per-group 4ch data = 4.8 GB, file = 9.6 GB. The full-parent DATA (4.8 GB/group)
is never read (proven by stage 5); the only large anonymous allocation is the
2.4 GB time vector, reloaded eagerly per BinaryFolderRecording reconstruction.

Run this twice: once on stock SI, once after editing baserecording.py:412 to
np.load(time_file, mmap_mode="r"). Peak USS should fall from ~time_vector x
(loads x workers) to ~the touched slice.
"""
import time
import threading
import shutil
import pathlib
import numpy as np
import psutil
import spikeinterface as si
import spikeinterface.sorters as ss
from spikeinterface.core import BinaryRecordingExtractor
from probeinterface import generate_linear_probe
from tetrode_analyses.sorting import sort_store  # noqa: F401

FS = 30000
NCH = 8
N = int(3e8)
NJOBS = 2
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize6")
TMP.mkdir(parents=True, exist_ok=True)
GB = 1e9


def peak_uss_during(fn):
    proc = psutil.Process()
    pk = {"u": 0}
    run = {"go": True}

    def loop():
        while run["go"]:
            u = 0
            for p in [proc] + proc.children(recursive=True):
                try:
                    u += p.memory_full_info().uss
                except psutil.Error:
                    pass
            pk["u"] = max(pk["u"], u)
            time.sleep(0.2)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    out = fn()
    run["go"] = False
    th.join(timeout=2)
    return out, pk["u"] / GB


raw = TMP / "raw.dat"
print(f"[{time.strftime('%T')}] building sparse {N*NCH*4/GB:.1f} GB binary ...", flush=True)
with open(raw, "wb") as f:
    f.seek(N * NCH * 4 - 1)
    f.write(b"\0")
mm = np.memmap(raw, dtype="float32", mode="r+", shape=(N, NCH))
mm[: 320 * FS] = (np.random.default_rng(0).standard_normal((320 * FS, NCH)) * 50).astype("f4")
mm.flush()
del mm
rec = BinaryRecordingExtractor(file_paths=[str(raw)], sampling_frequency=FS, num_channels=NCH, dtype="float32")
pr = generate_linear_probe(num_elec=NCH, ypitch=50)
pr.set_device_channel_indices(np.arange(NCH))
rec = rec.set_probe(pr)
rec.set_property("group", (np.arange(NCH) // 4).astype(int))
rec.set_times(np.arange(N, dtype="float64") / FS, with_warning=False)
print(f"[{time.strftime('%T')}] saving binary folder (time vector = {N*8/GB:.1f} GB)...", flush=True)
rec.save(folder=str(TMP / "folder"), overwrite=True, n_jobs=8, progress_bar=False)
del rec
raw.unlink(missing_ok=True)

KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=42,
          scheme2_training_duration_sec=300, scheme2_training_recording_sampling_mode="uniform",
          detect_threshold=5.5, detect_sign=-1)

loaded = si.load(str(TMP / "folder"))
sl = loaded.frame_slice(0, 300 * FS)


def run():
    sf = TMP / "sort"
    shutil.rmtree(sf, ignore_errors=True)
    return ss.run_sorter_by_property("mountainsort5", sl, grouping_property="group", folder=str(sf),
                                     engine="joblib", engine_kwargs={"n_jobs": NJOBS}, verbose=False, **KW)


try:
    srt, uss = peak_uss_during(run)
    print(f"\n=== run_sorter_by_property n_jobs={NJOBS}: {srt.get_num_units()} units, peak USS {uss:.2f} GB ===",
          flush=True)
    print(f"  one time vector = {N*8/GB:.2f} GB | full per-group 4ch data (never read) = {N*4*4/GB:.2f} GB", flush=True)
except Exception as e:
    import traceback
    print(f"\nERROR: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
finally:
    shutil.rmtree(TMP, ignore_errors=True)
print("DONE", flush=True)

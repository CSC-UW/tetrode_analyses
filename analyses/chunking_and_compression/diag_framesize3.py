"""Stage 3: is the 48h-anomaly caused by the full-length float64 TIME VECTOR?

Stages 1-2 ruled out frame count (tested to the exact 48h count, >2^32), 2^31/2^32,
split/property path, and channel count -- the synthetic stayed ~1.8 GB. The one
thing the synthetic lacked vs the real materialized binary: a full-length float64
time vector (5.2e9 x 8 B = 41.7 GB), which frame_slice carries at FULL length (the
same set_times quirk seen earlier), and which the per-crop fix dropped via
reset_times(). Test: same 48h 64-ch binary, but set a full time vector, then
frame_slice -> run_sorter_by_property, measure peak USS. Compare to the 1.76 GB
no-time-vector result from stage 2.
"""
import time
import threading
import shutil
import pathlib
import numpy as np
import psutil
import spikeinterface.sorters as ss
from spikeinterface.core import BinaryRecordingExtractor
from probeinterface import generate_linear_probe
from tetrode_analyses.sorting import sort_store  # noqa: F401

FS = 30000
NCH = 64
N = int(5.215e9)  # 48.3 h, the real frame count
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize3")
TMP.mkdir(parents=True, exist_ok=True)


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
            time.sleep(0.3)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    out = fn()
    run["go"] = False
    th.join(timeout=2)
    return out, pk["u"] / 1e9


binpath = TMP / "big.dat"
with open(binpath, "wb") as f:
    f.seek(N * NCH * 4 - 1)
    f.write(b"\0")
mm = np.memmap(binpath, dtype="float32", mode="r+", shape=(N, NCH))
rng = np.random.default_rng(0)
mm[: 320 * FS] = (rng.standard_normal((320 * FS, NCH)) * 50).astype("f4")
mm.flush()
del mm
rec = BinaryRecordingExtractor(file_paths=[str(binpath)], sampling_frequency=FS, num_channels=NCH, dtype="float32")
pr = generate_linear_probe(num_elec=NCH, ypitch=50)
pr.set_device_channel_indices(np.arange(NCH))
rec = rec.set_probe(pr)
rec.set_property("group", (np.arange(NCH) // 4).astype(int))

print(f"[{time.strftime('%T')}] allocating full-length time vector ({N*8/1e9:.0f} GB float64)...", flush=True)
rec.set_times(np.arange(N, dtype="float64") / FS)   # the 41.7 GB time vector
print(f"  has_time_vector={rec.has_time_vector(0)}", flush=True)
sl = rec.frame_slice(0, 300 * FS)
print(f"  slice frames={sl.get_num_frames()} | slice has_time_vector={sl.has_time_vector(0)}", flush=True)

KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=42,
          scheme2_training_duration_sec=300, scheme2_training_recording_sampling_mode="uniform",
          detect_threshold=5.5, detect_sign=-1)


def run():
    sf = TMP / "sort"
    shutil.rmtree(sf, ignore_errors=True)
    return ss.run_sorter_by_property("mountainsort5", sl, grouping_property="group", folder=str(sf),
                                     engine="joblib", engine_kwargs={"n_jobs": 5}, verbose=False, **KW)


try:
    srt, uss = peak_uss_during(run)
    print(f"\nRESULT with time vector: {srt.get_num_units()} units, peak USS {uss:.1f} GB "
          f"(stage-2 WITHOUT time vector was 1.76 GB)", flush=True)
except Exception as e:
    print(f"\nERROR with time vector: {type(e).__name__}: {e}", flush=True)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
print("DONE", flush=True)

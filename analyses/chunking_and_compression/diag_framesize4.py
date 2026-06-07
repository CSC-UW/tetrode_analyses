"""Stage 4: does the time vector get multiplied (parent + per-worker) in the real
file-backed BinaryFolderRecording + run_sorter_by_property path?

Stage 3 (in-memory time vector on a BinaryRecordingExtractor) showed ~42 GB = one
copy. The real run used a materialized BinaryFolderRecording (time vector in a FILE)
+ run_sorter_by_property (joblib workers). Test that path cheaply: 8 ch (2 groups),
1.5e9 frames -> 12 GB time vector. Save to a folder (file-backed times), reload,
frame_slice, run_sorter_by_property n_jobs=2, measure peak USS. If USS >> 12 GB,
the vector is materialized multiple times (parent + N workers) -> explains the ~500 GB.
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
N = int(1.5e9)   # 13.9 h -> 12 GB time vector
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize4")
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


# build a sparse binary + time vector, save to a FOLDER (file-backed times like the real materialize)
raw = TMP / "raw.dat"
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
rec.set_times(np.arange(N, dtype="float64") / FS)   # 12 GB
print(f"[{time.strftime('%T')}] saving to binary folder (file-backed times, {N*8/1e9:.0f} GB vector)...", flush=True)
rec.save(folder=str(TMP / "folder"), overwrite=True, n_jobs=8, progress_bar=False)
del rec
raw.unlink(missing_ok=True)

KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=42,
          scheme2_training_duration_sec=300, scheme2_training_recording_sampling_mode="uniform",
          detect_threshold=5.5, detect_sign=-1)

for tag, drop_times in [("with_times", False), ("reset_times", True)]:
    loaded = si.load(str(TMP / "folder"))
    sl = loaded.frame_slice(0, 300 * FS)
    if drop_times:
        sl.reset_times()
    print(f"\n=== {tag}: slice has_time_vector={sl.has_time_vector(0)} | time-vector size = {N*8/1e9:.0f} GB ===", flush=True)

    def run():
        sf = TMP / f"sort_{tag}"
        shutil.rmtree(sf, ignore_errors=True)
        return ss.run_sorter_by_property("mountainsort5", sl, grouping_property="group", folder=str(sf),
                                         engine="joblib", engine_kwargs={"n_jobs": 2}, verbose=False, **KW)
    try:
        srt, uss = peak_uss_during(run)
        print(f"  RESULT {tag}: {srt.get_num_units()} units, peak USS {uss:.1f} GB "
              f"(1 copy={N*8/1e9:.0f} GB; parent+2 workers ~{3*N*8/1e9:.0f} GB)", flush=True)
    except Exception as e:
        print(f"  ERROR {tag}: {type(e).__name__}: {e}", flush=True)
    finally:
        shutil.rmtree(TMP / f"sort_{tag}", ignore_errors=True)

shutil.rmtree(TMP, ignore_errors=True)
print("DONE", flush=True)

"""Stage 2 of the 48h-anomaly root-cause.

Stage 1 (diag_framesize.py) showed run_sorter (single, 4ch) on a frame-slice of a
>2^31-frame parent uses only ~0.25 GB -- frame_slice limits fine. The remaining
difference from the broken benchmark is run_sorter_by_PROPERTY (split_by group,
n_jobs=5) on a frame-slice of a LARGE multi-channel parent. The working per-crop
sweep used run_sorter_by_property too but on genuine crop binaries -> fine. So test:
run_sorter_by_property on a frame-slice of a 64-ch parent, at parent sizes straddling
2^31 frames. If the >2^31 case balloons (per-group ~ full-parent-of-4ch x n_jobs),
the culprit is split + large-parent (likely a 2^31 int32 site reached only via the
split/property path).
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
from tetrode_analyses.sorting import sort_store  # noqa: F401  (ms5 _segments patch)

FS = 30000
NCH = 64
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize2")
TMP.mkdir(parents=True, exist_ok=True)
THRESH = 2**32
# frames, hours: one well under 2^32, one just over (= the real 48h frame count)
CASES = [("under_2^32_37h", int(4.00e9), 37.0), ("over_2^32_48h", int(5.215e9), 48.3)]


def make_sparse_binary(path, n_frames):
    nbytes = n_frames * NCH * 4
    with open(path, "wb") as f:
        f.seek(nbytes - 1)
        f.write(b"\0")
    mm = np.memmap(path, dtype="float32", mode="r+", shape=(n_frames, NCH))
    rng = np.random.default_rng(0)
    mm[: 320 * FS] = (rng.standard_normal((320 * FS, NCH)) * 50).astype("f4")
    mm.flush()
    del mm
    rec = BinaryRecordingExtractor(file_paths=[str(path)], sampling_frequency=FS, num_channels=NCH, dtype="float32")
    pr = generate_linear_probe(num_elec=NCH, ypitch=50)
    pr.set_device_channel_indices(np.arange(NCH))
    rec = rec.set_probe(pr)
    rec.set_property("group", (np.arange(NCH) // 4).astype(int))  # 16 tetrode groups
    return rec


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


KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=42,
          scheme2_training_duration_sec=300, scheme2_training_recording_sampling_mode="uniform",
          detect_threshold=5.5, detect_sign=-1)

for label, n_frames, hours in CASES:
    binpath = TMP / f"{label}.dat"
    print(f"\n=== {label}: {n_frames} frames ({hours} h), {'>' if n_frames > THRESH else '<'} 2^31 | 64ch/16 groups ===", flush=True)
    rec = make_sparse_binary(binpath, n_frames)
    sl = rec.frame_slice(0, 300 * FS)
    full4ch_gb = n_frames * 4 * 4 / 1e9
    print(f"  parent frames={rec.get_num_frames()} | slice frames={sl.get_num_frames()} | full-4ch={full4ch_gb:.0f} GB", flush=True)

    def run():
        sf = TMP / f"sort_{label}"
        shutil.rmtree(sf, ignore_errors=True)
        return ss.run_sorter_by_property("mountainsort5", sl, grouping_property="group", folder=str(sf),
                                         engine="joblib", engine_kwargs={"n_jobs": 5}, verbose=False, **KW)

    try:
        srt, uss = peak_uss_during(run)
        print(f"  RESULT {label}: {srt.get_num_units()} units, peak USS {uss:.2f} GB "
              f"(small ~few GB if slice limits; ~{5*full4ch_gb:.0f} GB if 5 workers load full)", flush=True)
    except Exception as e:
        print(f"  ERROR {label}: {type(e).__name__}: {e}", flush=True)
    finally:
        binpath.unlink(missing_ok=True)
        shutil.rmtree(TMP / f"sort_{label}", ignore_errors=True)

shutil.rmtree(TMP, ignore_errors=True)
print("\nDONE", flush=True)

"""Stage 5: LOCALIZE the full-parent anonymous allocation in the
BinaryFolderRecording + frame_slice + sort path.

Single in-process run == exactly what ONE joblib worker does. Instruments:
  - numpy.load  -> logs every load > 100 MB with a short stack (catches the
    eager time-vector reload in _extra_metadata_from_folder)
  - BinaryRecordingSegment.get_traces -> records max single-read size and total
    frames read, and whether the returned array is a view (mmap, reclaimable)
    or an anonymous copy (counts as USS)

Cheap scale: NCH=8, N=3e8 (~2.8 h). time vector = 2.4 GB; per-group(4ch) data
= 4.8 GB; file = 9.6 GB (sparse, only 320 s of noise written).

If peak USS ~= one time vector (2.4 GB) and get_traces never returns a big copy,
the per-worker blowup is the time-vector np.load, NOT a data load.
"""
import time
import threading
import shutil
import pathlib
import traceback
import numpy as np
import psutil
import spikeinterface as si
import spikeinterface.sorters as ss
from spikeinterface.core import BinaryRecordingExtractor
import spikeinterface.core.binaryrecordingextractor as bre
from probeinterface import generate_linear_probe
from tetrode_analyses.sorting import sort_store  # noqa: F401  (ms5 _segments patch)

FS = 30000
NCH = 8
N = int(3e8)
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize5")
TMP.mkdir(parents=True, exist_ok=True)
GB = 1e9

# ---- instrumentation -------------------------------------------------------
np_load_log = []
_orig_np_load = np.load


def traced_np_load(*a, **k):
    arr = _orig_np_load(*a, **k)
    try:
        nbytes = arr.nbytes
    except AttributeError:
        nbytes = -1
    if nbytes > 100e6:
        stack = "".join(traceback.format_stack(limit=8)[:-1])
        np_load_log.append((str(a[0]) if a else "?", nbytes, k.get("mmap_mode"), stack))
    return arr


np.load = traced_np_load

gt_stats = {"max_frames": 0, "max_nbytes": 0, "n_calls": 0, "total_frames": 0, "big_copies": 0}
_orig_get_traces = bre.BinaryRecordingSegment.get_traces


def traced_get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
    out = _orig_get_traces(self, start_frame, end_frame, channel_indices)
    nframes = (end_frame or 0) - (start_frame or 0)
    gt_stats["n_calls"] += 1
    gt_stats["total_frames"] += nframes
    gt_stats["max_frames"] = max(gt_stats["max_frames"], nframes)
    gt_stats["max_nbytes"] = max(gt_stats["max_nbytes"], out.nbytes)
    # base buffer None => owns its data => anonymous copy; else => view (mmap)
    owns = out.base is None
    if owns and out.nbytes > 100e6:
        gt_stats["big_copies"] += 1
    return out


bre.BinaryRecordingSegment.get_traces = traced_get_traces


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


# ---- build a BinaryFolder WITH a full time vector --------------------------
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
rec.set_times(np.arange(N, dtype="float64") / FS, with_warning=False)  # 2.4 GB
print(f"[{time.strftime('%T')}] saving to binary folder (time vector = {N*8/GB:.1f} GB)...", flush=True)
rec.save(folder=str(TMP / "folder"), overwrite=True, n_jobs=8, progress_bar=False)
del rec
raw.unlink(missing_ok=True)

KW = dict(scheme="2", filter=False, whiten=True, whitening_seed=42,
          scheme2_training_duration_sec=300, scheme2_training_recording_sampling_mode="uniform",
          detect_threshold=5.5, detect_sign=-1)

# ---- ONE worker's worth of work, in-process -------------------------------
np_load_log.clear()
for k in gt_stats:
    gt_stats[k] = 0

loaded = si.load(str(TMP / "folder"))
print(f"\n[{time.strftime('%T')}] after si.load: has_time_vector={loaded.has_time_vector(0)} "
      f"num_frames={loaded.get_num_frames()}", flush=True)
sl = loaded.frame_slice(0, 300 * FS)
group0 = sl.split_by("group")[0]  # ChannelSlice(FrameSlice(BinaryFolder)) -- exactly one worker's recording
print(f"  group0 = {type(group0).__name__} frames={group0.get_num_frames()} "
      f"chans={group0.get_num_channels()} has_time_vector={group0.has_time_vector(0)}", flush=True)


def run():
    sf = TMP / "sort0"
    shutil.rmtree(sf, ignore_errors=True)
    return ss.run_sorter("mountainsort5", group0, folder=str(sf), remove_existing_folder=True, verbose=False, **KW)


try:
    srt, uss = peak_uss_during(run)
    print(f"\n=== RESULT: {srt.get_num_units()} units, peak USS {uss:.2f} GB ===", flush=True)
    print(f"  (one time vector = {N*8/GB:.2f} GB ; full per-group 4ch data = {N*4*4/GB:.2f} GB)", flush=True)
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()

print(f"\n--- numpy.load calls > 100 MB ({len(np_load_log)}): ---", flush=True)
for path, nbytes, mmap_mode, stack in np_load_log:
    print(f"  {nbytes/GB:.2f} GB  mmap_mode={mmap_mode}  {pathlib.Path(path).name}", flush=True)
    print("    " + stack.replace("\n", "\n    ").rstrip(), flush=True)

print(f"\n--- BinaryRecordingSegment.get_traces stats ---", flush=True)
print(f"  n_calls={gt_stats['n_calls']}  max_single_read={gt_stats['max_frames']} frames "
      f"({gt_stats['max_nbytes']/GB:.3f} GB)  total_frames_read={gt_stats['total_frames']} "
      f"(={gt_stats['total_frames']/FS:.0f} s)  big_anonymous_copies={gt_stats['big_copies']}", flush=True)

shutil.rmtree(TMP, ignore_errors=True)
print("\nDONE", flush=True)

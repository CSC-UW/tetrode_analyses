"""Root-cause the 48h-binary memory anomaly: is it a 2^31-frame (int32) threshold?

Hypothesis: a recording with > 2^31 = 2,147,483,648 frames trips an int32 overflow
in SI's frame handling, so frame_slice(0, 300 s) no longer limits what ms5 reads ->
it loads the whole recording. Test with 4-channel sparse binaries straddling 2^31
frames (cheap: ~35 GB each, not 1.3 TB), frame-slice to 300 s, sort, measure peak USS.
If the >2^31 case balloons and the <2^31 case stays small (~1 GB), hypothesis confirmed.
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
from tetrode_analyses.sorting import sort_store  # noqa: F401  (applies ms5 _segments patch)

FS = 30000
NCH = 4
TMP = pathlib.Path("/nvme/neuropixels/tmp/diag_framesize")
TMP.mkdir(parents=True, exist_ok=True)
THRESH = 2**31
CASES = [("under_2^31", int(2.00e9), 18.5), ("over_2^31", int(2.20e9), 20.4)]  # frames, hours


def make_sparse_binary(path, n_frames):
    nbytes = n_frames * NCH * 4  # float32
    with open(path, "wb") as f:           # sparse file, full apparent size, ~instant
        f.seek(nbytes - 1)
        f.write(b"\0")
    # write real noise into just the first 320 s so the 300 s slice has spikes
    mm = np.memmap(path, dtype="float32", mode="r+", shape=(n_frames, NCH))
    rng = np.random.default_rng(0)
    mm[: 320 * FS] = (rng.standard_normal((320 * FS, NCH)) * 50).astype("f4")
    mm.flush()
    del mm
    rec = BinaryRecordingExtractor(file_paths=[str(path)], sampling_frequency=FS, num_channels=NCH, dtype="float32")
    pr = generate_linear_probe(num_elec=NCH, ypitch=300)
    pr.set_device_channel_indices(np.arange(NCH))
    return rec.set_probe(pr)


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
    print(f"\n=== {label}: {n_frames} frames ({hours} h), {'>' if n_frames > THRESH else '<'} 2^31 ===", flush=True)
    rec = make_sparse_binary(binpath, n_frames)
    print(f"  parent get_num_frames()={rec.get_num_frames()} (correct={n_frames})", flush=True)
    sl = rec.frame_slice(0, 300 * FS)
    print(f"  frame_slice get_num_frames()={sl.get_num_frames()} (expected {300*FS})", flush=True)

    def run():
        sf = TMP / f"sort_{label}"
        shutil.rmtree(sf, ignore_errors=True)
        return ss.run_sorter("mountainsort5", sl, folder=str(sf), remove_existing_folder=True, verbose=False, **KW)

    try:
        srt, uss = peak_uss_during(run)
        print(f"  RESULT {label}: {srt.get_num_units()} units, peak USS {uss:.2f} GB "
              f"(expect ~1 GB if frame_slice limits; ~{n_frames*NCH*4/1e9:.0f} GB if it loads full)", flush=True)
    except Exception as e:
        print(f"  ERROR {label}: {type(e).__name__}: {e}", flush=True)
    finally:
        binpath.unlink(missing_ok=True)
        shutil.rmtree(TMP / f"sort_{label}", ignore_errors=True)

shutil.rmtree(TMP, ignore_errors=True)
print("\nDONE", flush=True)

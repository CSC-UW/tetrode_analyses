"""Extended WavPack bps sweep for the README compression table.

Regenerates the `bps` table (ratio / projected size / in-band err / err-vs-noise /
max abs err) over a wider range of bps, to assess whether a higher-fidelity lossy
setting is acceptable for production now that bps=2.25 sorting agreement was judged
too low.

Methodology (matches 04_/05_, unified onto the production chunked store):
- ratio + max abs err: WavPack zarr store, chunks (30 s, 4 ch) = the tetrode
  grouping actually used in production (channel grouping drives WavPack's ratio).
- in-band err: 300-6000 Hz band of the (RAW - decoded) residual, per-channel RMS,
  median over channels, in uV (bit_volts = 0.195).
- noise floor: in-band robust noise (MAD->std) of RAW itself, median over channels.
- 667 GB -> : the full Open Ephys flat-binary total divided by the measured ratio.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import zarr
from scipy.signal import butter, sosfiltfilt
from wavpack_numcodecs import WavPack

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")  # (N, 64) int16
RAW_f4 = RAW.astype("f4")
N, NCH = RAW.shape
FS = 30000
BITVOLTS = 0.1949999928
TC, CC = 30 * FS, 4
FULL_GB = 667.0
SCRATCH = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/stores_ext")
SCRATCH.mkdir(parents=True, exist_ok=True)
RAW_BYTES = RAW.nbytes

BPS = [0.0, 2.25, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0]  # 0 = lossless

sos = butter(3, [300, 6000], btype="band", fs=FS, output="sos")
def bandpass(x):
    return sosfiltfilt(sos, x, axis=0)

# in-band robust noise floor of RAW (per channel -> median), uV
raw_bp = bandpass(RAW_f4)
noise_uV = (np.median(np.abs(raw_bp), 0) / 0.6745) * BITVOLTS
noise_floor = float(np.median(noise_uV))
print(f"in-band (300-6000 Hz) noise floor: median {noise_floor:.2f} uV "
      f"(range {noise_uV.min():.2f}-{noise_uV.max():.2f})", flush=True)

out = []
for bps in BPS:
    t0 = time.perf_counter()
    path = SCRATCH / f"wp_bps{bps}.zarr"
    if path.exists():
        shutil.rmtree(path)
    comp = WavPack(bps=bps) if bps > 0 else WavPack()
    store = zarr.DirectoryStore(str(path))
    z = zarr.create(shape=(N, NCH), chunks=(TC, CC), dtype="<i2", compressor=comp, store=store)
    z[:] = RAW
    store.close()
    disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    dec = zarr.open(str(path), mode="r")[:]
    shutil.rmtree(path)

    lossless = bool(np.array_equal(dec, RAW))
    ratio = RAW_BYTES / disk
    resid = RAW_f4 - dec.astype("f4")
    max_abs_uV = float(np.abs(resid).max()) * BITVOLTS
    if lossless:
        inband_uV = 0.0
        err_over_noise = 0.0
    else:
        resid_bp = bandpass(resid)
        inband_per_ch = resid_bp.std(0) * BITVOLTS
        inband_uV = float(np.median(inband_per_ch))
        err_over_noise = float(np.median(inband_per_ch / noise_uV))
    rec = dict(bps=bps, lossless=lossless, ratio=round(ratio, 2),
               projected_GB=round(FULL_GB / ratio, 1),
               inband_err_uV=round(inband_uV, 2), err_over_noise=round(err_over_noise, 3),
               max_abs_err_uV=round(max_abs_uV, 1), secs=round(time.perf_counter() - t0, 1))
    out.append(rec)
    print(f"bps={bps:<5} ratio={rec['ratio']:5.2f}x  ~{rec['projected_GB']:6.1f}GB  "
          f"inband={rec['inband_err_uV']:5.2f}uV  err/noise={rec['err_over_noise']:.3f}  "
          f"maxabs={rec['max_abs_err_uV']:6.1f}uV  ({rec['secs']}s)", flush=True)

result = dict(noise_floor_uV=round(noise_floor, 2), bit_volts=BITVOLTS,
              sample_shape=[int(N), int(NCH)], full_GB=FULL_GB, results=out)
pathlib.Path("/nvme/neuropixels/tmp/cc_bench/results_bps_ext.json").write_text(json.dumps(result, indent=2))

# emit a ready-to-paste markdown table
def fmt(r):
    if r["lossless"]:
        return f"| 0 (lossless) | {r['ratio']:.2f}× | ~{r['projected_GB']:.0f} GB | 0 | 0 | 0 |"
    return (f"| {r['bps']:g} | {r['ratio']:.2f}× | ~{r['projected_GB']:.0f} GB | "
            f"{r['inband_err_uV']:.2f} µV | {r['err_over_noise']:.2f}× | {r['max_abs_err_uV']:.0f} µV |")

print("\n--- MARKDOWN TABLE ---", flush=True)
print("| `bps` | ratio | 667 GB → | in-band err | err / noise floor | max abs err |", flush=True)
print("|---|---|---|---|---|---|", flush=True)
for r in out:
    print(fmt(r), flush=True)
print("DONE", flush=True)

"""WavPack lossless vs hybrid-lossy (bps) on the tetrode sample.

Quantifies compression ratio AND the error introduced, so the lossy-vs-lossless
choice is evidence-based. Error reported in int16 LSB and microvolts
(bit_volts = 0.195 uV) and relative to the signal's own std.
"""
import json, time, shutil, pathlib
import numpy as np
import zarr
from wavpack_numcodecs import WavPack

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")  # (N,64) int16
N, NCH = RAW.shape
FS = 30000; BITVOLTS = 0.1949999928
TC, CC = 30*FS, 4
SCRATCH = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/stores3"); SCRATCH.mkdir(parents=True, exist_ok=True)
RAW_BYTES = RAW.nbytes
sig_std = RAW.astype("f4").std()

# per-channel std (noise) for context
ch_std = RAW.astype("f4").std(0)

BPS = [0.0, 2.25, 3.0, 3.5, 4.0]  # 0 = lossless
out = []
for bps in BPS:
    path = SCRATCH/f"wp_bps{bps}.zarr"
    if path.exists(): shutil.rmtree(path)
    comp = WavPack(bps=bps) if bps > 0 else WavPack()
    store = zarr.DirectoryStore(str(path))
    z = zarr.create(shape=(N,NCH), chunks=(TC,CC), dtype="<i2", compressor=comp, store=store)
    z[:] = RAW
    store.close()
    disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    dec = zarr.open(str(path), mode="r")[:]
    resid = RAW.astype("f4") - dec.astype("f4")
    rms = float(np.sqrt((resid**2).mean()))
    maxerr = float(np.abs(resid).max())
    rec = dict(bps=bps, lossless=bool(np.array_equal(dec, RAW)),
               ratio=round(RAW_BYTES/disk, 3), disk_MB=round(disk/1e6, 1),
               resid_rms_LSB=round(rms, 4), resid_rms_uV=round(rms*BITVOLTS, 4),
               max_abs_err_LSB=round(maxerr, 1),
               resid_rms_pct_of_signal=round(100*rms/sig_std, 4))
    out.append(rec)
    print(f"bps={bps:<5} ratio={rec['ratio']:.2f} disk={rec['disk_MB']:6.1f}MB "
          f"lossless={rec['lossless']} resid_rms={rec['resid_rms_uV']:.4f}uV "
          f"({rec['resid_rms_pct_of_signal']:.3f}% of signal std) maxerr={rec['max_abs_err_LSB']}LSB", flush=True)
    shutil.rmtree(path)

pathlib.Path("/nvme/neuropixels/tmp/cc_bench/results3.json").write_text(json.dumps(
    dict(signal_std_LSB=round(float(sig_std),1), signal_std_uV=round(float(sig_std)*BITVOLTS,2),
         per_channel_std_LSB_min=round(float(ch_std.min()),1), per_channel_std_LSB_max=round(float(ch_std.max()),1),
         bit_volts=BITVOLTS, results=out), indent=2))
print("DONE", flush=True)

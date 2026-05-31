"""Compressor x chunk-shape benchmark for tetrode spikesorting zarr conversion.

Metrics per (compressor, chunk-shape):
  - ratio: raw_bytes / on-disk_bytes (higher = better)
  - write_MBps: single-core encode throughput on the raw int16 sample
  - group_read_MBps: throughput reading data the way per-group sorting does
        (each of 16 tetrodes read once, in 1 s time windows, 4 ch at a time);
        decompressed MB / wall seconds. Sensitive to read amplification when
        the channel chunk is wider than a tetrode.
  - lossless: roundtrip identity check on a spot chunk.

Single-threaded codecs for fair per-core comparison (all parallelize in prod).
"""
import json, time, shutil, pathlib
import numpy as np
import zarr
import numcodecs
from numcodecs import Blosc, Zstd, Shuffle
from wavpack_numcodecs import WavPack

numcodecs.blosc.set_nthreads(1)

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")  # (N, 64) int16
N, NCH = RAW.shape
FS = 30000
TT = 4                      # channels per tetrode
N_TT = NCH // TT
WIN = FS                    # 1 s read window, mimics get_traces calls
SCRATCH = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/stores")
SCRATCH.mkdir(parents=True, exist_ok=True)
RAW_BYTES = RAW.nbytes

def wavpack():
    return WavPack(num_encoding_threads=1, num_decoding_threads=1)

COMPRESSORS = {
    "blosc-zstd-l5-bitshuffle": (Blosc("zstd", 5, Blosc.BITSHUFFLE), None),
    "blosc-zstd-l5-shuffle":    (Blosc("zstd", 5, Blosc.SHUFFLE), None),
    "blosc-zstd-l9-bitshuffle": (Blosc("zstd", 9, Blosc.BITSHUFFLE), None),
    "blosc-lz4-l5-bitshuffle":  (Blosc("lz4", 5, Blosc.BITSHUFFLE), None),
    "zstd-l5-shuffle":          (Zstd(level=5), [Shuffle(elementsize=2)]),
    "wavpack-l1":               (wavpack(), None),
}

# (time_chunk, channel_chunk)
SHAPES = {
    "t1s_c4":   (1*FS, 4),
    "t3s_c4":   (3*FS, 4),
    "t10s_c4":  (10*FS, 4),
    "t30s_c4":  (30*FS, 4),
    "t3s_c64":  (3*FS, 64),
    "t10s_c64": (10*FS, 64),
}

def group_read(z):
    """Read each tetrode once in 1 s windows; return (MBps, seconds, bytes)."""
    t0 = time.perf_counter()
    total = 0
    for tt in range(N_TT):
        c0 = tt * TT
        for s in range(0, N, WIN):
            blk = z[s:s+WIN, c0:c0+TT]
            total += blk.nbytes
    dt = time.perf_counter() - t0
    return total/1e6/dt, dt, total

results = []
for cname, (comp, filters) in COMPRESSORS.items():
    for sname, (tc, cc) in SHAPES.items():
        path = SCRATCH / f"{cname}__{sname}.zarr"
        if path.exists():
            shutil.rmtree(path)
        store = zarr.DirectoryStore(str(path))
        t0 = time.perf_counter()
        z = zarr.create(shape=(N, NCH), chunks=(tc, cc), dtype="<i2",
                        compressor=comp, filters=filters, store=store)
        z[:] = RAW
        write_dt = time.perf_counter() - t0
        store.close()
        # on-disk size
        disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        ratio = RAW_BYTES / disk
        # lossless check (spot: first 3 s, all ch)
        zr = zarr.open(str(path), mode="r")
        ok = np.array_equal(zr[:3*FS, :], RAW[:3*FS, :])
        # group read (drop OS-cache effect is minor; we care about decode+amplification)
        gr_mbps, gr_dt, gr_bytes = group_read(zr)
        rec = dict(compressor=cname, shape=sname, time_chunk=tc, chan_chunk=cc,
                   ratio=round(ratio, 3), disk_MB=round(disk/1e6, 1),
                   write_MBps=round(RAW_BYTES/1e6/write_dt, 1),
                   group_read_MBps=round(gr_mbps, 1),
                   read_amplification=round(cc / TT, 2),  # ch decompressed per tetrode / 4
                   lossless=bool(ok))
        results.append(rec)
        print(f"{cname:26s} {sname:9s} ratio={rec['ratio']:.2f} "
              f"w={rec['write_MBps']:6.1f} gr={rec['group_read_MBps']:7.1f}MB/s "
              f"amp={rec['read_amplification']:.2f} ll={ok}")
        shutil.rmtree(path)  # keep scratch small

out = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/results.json")
out.write_text(json.dumps(results, indent=2))
print("wrote", out)

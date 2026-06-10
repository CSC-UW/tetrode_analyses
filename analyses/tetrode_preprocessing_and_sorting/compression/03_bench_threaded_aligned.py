"""Round 2: production-realistic read throughput for the finalists.

Round 1 ran codecs single-threaded with a fixed 1 s read window. Here we:
  - use production threading (blosc multi-thread; WavPack default 8 decode threads)
  - read each tetrode chunk-aligned (read block == zarr time chunk) so every
    chunk is decoded exactly once -> isolates true decode throughput and the
    channel-amplification effect, not the window<chunk re-decode artifact.

Writes results incrementally to results2.jsonl (kill-safe).
"""
import json, time, shutil, pathlib
import numpy as np
import zarr, numcodecs
from numcodecs import Blosc
from wavpack_numcodecs import WavPack

numcodecs.blosc.set_nthreads(8)

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")
N, NCH = RAW.shape
FS = 30000; TT = 4; N_TT = NCH // TT
SCRATCH = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/stores2"); SCRATCH.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/results2.jsonl"); OUT.write_text("")
RAW_BYTES = RAW.nbytes

def wp():  # default num_decoding_threads=8
    return WavPack(num_encoding_threads=4)

COMP = {
    "blosc-zstd-l5-bitshuffle": (Blosc("zstd", 5, Blosc.BITSHUFFLE), None),
    "wavpack-l1":               (wp(), None),
}
SHAPES = {  # (time_chunk, chan_chunk)
    "t1s_c4":  (1*FS, 4),
    "t10s_c4": (10*FS, 4),
    "t30s_c4": (30*FS, 4),
    "t30s_c64":(30*FS, 64),
}

def aligned_group_read(z, tc):
    """Each tetrode read once, in blocks == time chunk (one decode/chunk)."""
    t0 = time.perf_counter(); total = 0
    for tt in range(N_TT):
        c0 = tt*TT
        for s in range(0, N, tc):
            total += z[s:s+tc, c0:c0+TT].nbytes
    dt = time.perf_counter()-t0
    return total/1e6/dt, dt

for cname,(comp,filt) in COMP.items():
    for sname,(tc,cc) in SHAPES.items():
        path = SCRATCH/f"{cname}__{sname}.zarr"
        if path.exists(): shutil.rmtree(path)
        store = zarr.DirectoryStore(str(path))
        t0 = time.perf_counter()
        z = zarr.create(shape=(N,NCH), chunks=(tc,cc), dtype="<i2", compressor=comp, filters=filt, store=store)
        z[:] = RAW
        wdt = time.perf_counter()-t0
        store.close()
        disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        zr = zarr.open(str(path), mode="r")
        # warm read (fills OS cache) then timed read -> measures decode, not disk
        aligned_group_read(zr, tc)
        rmbps, rdt = aligned_group_read(zr, tc)
        rec = dict(compressor=cname, shape=sname, time_chunk=tc, chan_chunk=cc,
                   ratio=round(RAW_BYTES/disk,3), disk_MB=round(disk/1e6,1),
                   write_MBps=round(RAW_BYTES/1e6/wdt,1),
                   aligned_group_read_MBps=round(rmbps,1),
                   chan_amplification=cc//TT)
        with OUT.open("a") as f: f.write(json.dumps(rec)+"\n")
        print(f"{cname:26s} {sname:9s} ratio={rec['ratio']:.2f} "
              f"w={rec['write_MBps']:7.1f} read={rec['aligned_group_read_MBps']:7.1f}MB/s "
              f"camp={cc//TT}", flush=True)
        shutil.rmtree(path)
print("DONE", flush=True)

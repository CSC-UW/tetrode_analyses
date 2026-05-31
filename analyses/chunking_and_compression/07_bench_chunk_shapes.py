"""Chunk-shape benchmark: channel_chunk 1 vs 4 vs 64, controlling for chunk size.

Closes two gaps in the earlier matrix (02_/03_): channel_chunk=1 was never
tested, and c4-vs-c64 were only compared at equal *time length*, not equal
*total chunk bytes*. Two families:

  sm_*  : size-matched   -> ~2.4 MB raw/chunk for all of c1/c4/c64 (equal total
          bytes; isolates the channel-shape effect on ratio & read speed).
  et_*  : equal time len -> 30 s for all (the natural-use comparison; chunk
          bytes scale with channel count).

Production threading; chunk-aligned per-tetrode-group reads (each chunk decoded
once). Kill-safe JSONL output. Run via the workspace env.
"""
import json, time, shutil, pathlib
import numpy as np
import zarr, numcodecs
from numcodecs import Blosc
from wavpack_numcodecs import WavPack

numcodecs.blosc.set_nthreads(8)

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")  # (N,64) int16
N, NCH = RAW.shape
FS = 30000; TT = 4; N_TT = NCH // TT
SCRATCH = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/stores_shapes"); SCRATCH.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/results_chunk_shapes.jsonl"); OUT.write_text("")
RAW_BYTES = RAW.nbytes

# size-matched family: ~1.2M elements/chunk = 2.4 MB raw, realised three ways
SHAPES = {
    "sm_c1_t40s":   (1_200_000, 1),
    "sm_c4_t10s":   (300_000, 4),
    "sm_c64_t625ms":(18_750, 64),
    "et_c1_t30s":   (900_000, 1),
    "et_c4_t30s":   (900_000, 4),
    "et_c64_t30s":  (900_000, 64),
}
COMP = {
    "blosc-zstd-l5-bitshuffle": Blosc("zstd", 5, Blosc.BITSHUFFLE),
    "wavpack-bps2.25":          WavPack(bps=2.25),
}

def aligned_group_read(z, tc):
    """Each tetrode read once, in blocks == time chunk (one decode/chunk)."""
    t0 = time.perf_counter(); total = 0
    for tt in range(N_TT):
        c0 = tt * TT
        for s in range(0, N, tc):
            total += z[s:s+tc, c0:c0+TT].nbytes
    return total/1e6/(time.perf_counter()-t0)

def n_total_chunks(tc, cc):
    return -(-N // tc) * -(-NCH // cc)   # ceil-div product

for cname, comp in COMP.items():
    for sname, (tc, cc) in SHAPES.items():
        path = SCRATCH / f"{cname}__{sname}.zarr"
        if path.exists(): shutil.rmtree(path)
        store = zarr.DirectoryStore(str(path))
        t0 = time.perf_counter()
        z = zarr.create(shape=(N, NCH), chunks=(tc, cc), dtype="<i2", compressor=comp, store=store)
        z[:] = RAW
        wdt = time.perf_counter() - t0
        store.close()
        disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        n_files = sum(1 for p in path.rglob("*") if p.is_file())
        zr = zarr.open(str(path), mode="r")
        aligned_group_read(zr, tc)                       # warm
        rmbps = aligned_group_read(zr, tc)
        rec = dict(compressor=cname, shape=sname, time_chunk=tc, chan_chunk=cc,
                   chunk_MB_raw=round(tc*cc*2/1e6, 2),
                   ratio=round(RAW_BYTES/disk, 3), disk_MB=round(disk/1e6, 1),
                   write_MBps=round(RAW_BYTES/1e6/wdt, 1),
                   aligned_group_read_MBps=round(rmbps, 1),
                   chan_amplification=cc // TT if cc >= TT else None,
                   n_chunk_files=n_files,
                   files_per_exp1_est=int(n_total_chunks(tc, cc) * (3_025_748_403 / N)))
        with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
        print(f"{cname:26s} {sname:14s} chunk={rec['chunk_MB_raw']:6.2f}MB "
              f"ratio={rec['ratio']:.3f} w={rec['write_MBps']:7.1f} "
              f"read={rec['aligned_group_read_MBps']:8.1f}MB/s "
              f"files/exp1~{rec['files_per_exp1_est']:,}", flush=True)
        shutil.rmtree(path)
print("DONE", flush=True)

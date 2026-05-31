import time, numpy as np, pathlib
SRC = "/Volumes/neuropixel_archive/tetrode_data/2026-05-27_09-07-52/Record Node 121/experiment1/recording1/continuous/Acquisition_Board-120.acquisition_board/continuous.dat"
NCH = 64; FS = 30000; SEG_S = 60
OUT = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")
mm = np.memmap(SRC, dtype="<i2", mode="r")
n_total = mm.shape[0] // NCH
mm = mm.reshape(n_total, NCH)
print(f"total samples={n_total:,} ({n_total/FS/3600:.2f} h)")
seg = SEG_S * FS
# 3 segments: 10%, 50%, 85% through the recording
starts = [int(n_total*f) for f in (0.10, 0.50, 0.85)]
chunks = []
for s in starts:
    t0 = time.perf_counter()
    block = np.array(mm[s:s+seg])   # force read from disk
    dt = time.perf_counter()-t0
    mb = block.nbytes/1e6
    print(f"seg@{s:>13,}: {block.shape} {mb:.0f}MB in {dt:.2f}s = {mb/dt:.0f} MB/s")
    chunks.append(block)
out = np.concatenate(chunks, axis=0)
np.save(OUT, out)
print(f"saved {OUT} shape={out.shape} {out.nbytes/1e6:.0f}MB")
# quick stats
print("dtype", out.dtype, "min", out.min(), "max", out.max(), "std", out.std().round(1))

"""Convert the full 2026-05-27_09-07-52 session (both experiments) to both stores.

get_recording once (concatenated, session-relative sync clock, probegroup) ->
convert_recording to wavpack-bps2.25 and blosc-zstd. Times + reports sizes.
"""
import json, shutil, time, pathlib
import numpy as np
from tetrode_analyses.spikeinterface import get_recording, convert_recording

SESSION = "/Volumes/neuropixel_archive/tetrode_data/2026-05-27_09-07-52"
OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
OUT.mkdir(parents=True, exist_ok=True)
NJOBS = 32
SRC_BYTES = 5215033052 * 64 * 2  # full session traces, int16

print(f"[{time.strftime('%T')}] get_recording (full session)...", flush=True)
t0 = time.perf_counter()
rec, slice_table = get_recording(SESSION)
print(f"  loaded in {time.perf_counter()-t0:.0f}s | {rec.get_num_channels()}ch "
      f"{rec.get_num_frames()/rec.sampling_frequency/3600:.2f}h | "
      f"has_time_vector={rec.has_time_vector(0)}", flush=True)
slice_table.to_csv(OUT / "slice_table.csv", index=False)

results = []
for name, kw in [("wavpack-bps2.25", dict(compressor="wavpack", bps=2.25)),
                 ("blosc-zstd", dict(compressor="blosc-zstd"))]:
    dst = OUT / f"2026-05-27_09-07-52.{name}.zarr"
    if dst.exists():
        shutil.rmtree(dst)
    print(f"[{time.strftime('%T')}] convert {name} -> {dst}", flush=True)
    t0 = time.perf_counter()
    convert_recording(rec, dst, n_jobs=NJOBS, **kw)
    secs = time.perf_counter() - t0
    disk = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    rec_ = dict(store=name, seconds=round(secs, 1), minutes=round(secs / 60, 1),
                out_GB=round(disk / 1e9, 1), ratio=round(SRC_BYTES / disk, 3),
                source_read_MBps=round(SRC_BYTES / 1e6 / secs, 1))
    results.append(rec_)
    print(f"[{time.strftime('%T')}] DONE {name}: {rec_['minutes']} min | "
          f"{rec_['out_GB']} GB | ratio {rec_['ratio']}x | {rec_['source_read_MBps']} MB/s",
          flush=True)
    print("RESULT " + json.dumps(rec_), flush=True)

(OUT / "conversion_results.json").write_text(json.dumps(results, indent=2))
print("ALL DONE", flush=True)

"""De-risk: materialize + sort ONE mid-recording chunk; confirm per-worker memory
stays chunk-sized at start>0.

The frame_slice time-vector memory pitfall
(project_si_frame_slice_timevector_memory) blows worker RAM up to the FULL parent
when run_sorter_by_property reconstructs a frame_sliced BinaryFolderRecording.
`tracking.materialize_chunk` avoids it by slicing the *zarr* and saving a genuine
crop with reset_times(). Script 23 proved this for crops starting at frame 0; this
checks a crop starting at ~24 h (start_frame > 0), the regime the tracker uses.

PASS = peak USS is per-chunk scale (tens of GB, like the 23_ training-duration
table) rather than full-48 h scale (~415 GB).
"""
import json
import pathlib
import shutil
import threading
import time

import psutil
import spikeinterface as si

from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
WORK = SR / "_track_derisk"
OUTDIR = pathlib.Path(__file__).resolve().parent

FS = 30000.0
CHUNK_S = 1800.0
START_S = 86400.0  # ~24 h into the recording (well past frame 0)


class TreeSampler:
    """Peak USS/RSS/CPU over the process tree (USS excludes shared mmap cache)."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.proc = psutil.Process()
        self._run = False
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0

    def _loop(self):
        while self._run:
            try:
                procs = [self.proc] + self.proc.children(recursive=True)
            except psutil.Error:
                procs = [self.proc]
            uss = rss = 0
            cpu = 0.0
            for p in procs:
                try:
                    mi = p.memory_full_info()
                    uss += mi.uss
                    rss += mi.rss
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    pass
            self.peak_uss = max(self.peak_uss, uss)
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_cpu = max(self.peak_cpu, cpu)
            time.sleep(self.interval)

    def start(self):
        for p in [self.proc] + self.proc.children(recursive=True):
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        self._t.join(timeout=5)


def main():
    si.set_global_job_kwargs(n_jobs=5, progress_bar=False, chunk_duration="1s")
    WORK.mkdir(parents=True, exist_ok=True)
    bin_dir = WORK / "chunk_bin"
    sort_dir = WORK / "chunk_sort"
    shutil.rmtree(bin_dir, ignore_errors=True)
    shutil.rmtree(sort_dir, ignore_errors=True)

    nfr = si.read_zarr(str(BLOSC)).get_num_frames()
    chunk = tk.Chunk(
        index=0,
        start_frame=int(START_S * FS),
        end_frame=int((START_S + CHUNK_S) * FS),
        fs=FS,
    )
    print(f"[{time.strftime('%T')}] chunk [{chunk.t_start_s:.0f}-{chunk.t_end_s:.0f}s] "
          f"start_frame={chunk.start_frame} / {nfr} | /nvme free {shutil.disk_usage('/nvme').free/1e9:.0f} GB",
          flush=True)

    sampler = TreeSampler(interval=1.0)
    sampler.start()
    t0 = time.perf_counter()
    cb = tk.materialize_chunk(BLOSC, chunk, bin_dir, cmr="global", materialize_dtype="float32", n_jobs=96)
    mat_min = (time.perf_counter() - t0) / 60
    uss_after_mat = sampler.peak_uss
    print(f"[{time.strftime('%T')}] materialized in {mat_min:.1f} min | "
          f"peak USS so far {uss_after_mat/1e9:.1f} GB | binary {sum(f.stat().st_size for f in bin_dir.rglob('*') if f.is_file())/1e9:.1f} GB",
          flush=True)

    t1 = time.perf_counter()
    srt = tk.sort_chunk(cb, sort_dir, sort_n_jobs=5)
    sort_min = (time.perf_counter() - t1) / 60
    sampler.stop()

    n_units = srt.get_num_units()
    summary = {
        "chunk_start_s": START_S,
        "chunk_s": CHUNK_S,
        "start_frame": chunk.start_frame,
        "materialize_min": round(mat_min, 1),
        "sort_min": round(sort_min, 1),
        "n_units": int(n_units),
        "peak_uss_gb": round(sampler.peak_uss / 1e9, 1),
        "peak_rss_gb": round(sampler.peak_rss / 1e9, 1),
        "peak_cpu_pct": round(sampler.peak_cpu, 0),
        "binary_gb": round(sum(f.stat().st_size for f in bin_dir.rglob("*") if f.is_file()) / 1e9, 1),
        "full_parent_gb_reference": 415.0,
        "verdict": "PASS (per-chunk scale)" if sampler.peak_uss / 1e9 < 150 else "FAIL (full-parent scale)",
    }
    (OUTDIR / "derisk_chunk_uss.json").write_text(json.dumps(summary, indent=2))
    print("RESULT " + json.dumps(summary), flush=True)

    shutil.rmtree(bin_dir, ignore_errors=True)
    shutil.rmtree(sort_dir, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

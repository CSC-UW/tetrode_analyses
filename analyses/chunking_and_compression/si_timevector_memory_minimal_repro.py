"""Minimal reproduction: BinaryFolderRecording eagerly loads the full cached
time vector into anonymous RAM on every reconstruction.

Effect: with run_sorter_by_property (joblib), the recording is reconstructed
once per worker, so peak memory scales with (full-length time vector) x
(n_workers) -- even when the recording is a short frame_slice. No traces are
ever read; the cost is purely the time vector.

Run: python si_timevector_memory_minimal_repro.py
Needs only: spikeinterface, numpy, psutil  (no sorter).
"""
import shutil, pathlib, numpy as np, psutil, spikeinterface as si
from spikeinterface.core import NumpyRecording

GB = 1e9
N = int(2e8)           # 200 M samples -> 1.6 GB float64 time vector
FOLDER = pathlib.Path("/nvme/neuropixels/tmp/si_tv_repro")
shutil.rmtree(FOLDER, ignore_errors=True)
uss = lambda: psutil.Process().memory_full_info().uss / GB

# tiny traces, but a full-length time vector (this is what set_times persists)
rec = NumpyRecording([np.zeros((N, 1), dtype="float32")], sampling_frequency=30000)
rec.set_times(np.arange(N, dtype="float64") / 30000, with_warning=False)
rec.save(folder=FOLDER)          # writes traces_cached + times_cached_seg0.npy
tv_gb = (FOLDER / "times_cached_seg0.npy").stat().st_size / GB

base = uss()
reloaded = si.load(FOLDER)       # <-- reconstructs BinaryFolderRecording
after_load = uss()
sliced = reloaded.frame_slice(0, 30000)   # 1 s slice -- want almost no memory
after_slice = uss()

print(f"time vector on disk:           {tv_gb:.2f} GB")
print(f"USS after si.load(folder):     +{after_load - base:.2f} GB  "
      f"(stock SI ~= full time vector; mmap fix ~= 0)")
print(f"USS after frame_slice(0, 1 s): +{after_slice - base:.2f} GB")
print(f"frame_slice num_frames:        {sliced.get_num_frames()} (sort only needs these)")
print("\nCall site: BinaryFolderRecording.__init__ -> load_metadata_from_folder")
print("  -> baserecording._extra_metadata_from_folder: np.load(time_file)  [no mmap_mode]")

shutil.rmtree(FOLDER, ignore_errors=True)

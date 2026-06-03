"""Sort ONLY the wavpack store (blosc already done). Lower parallelism for the
shared-disk constraint. See 10_sort_ms5.py for the full two-store version."""
import json
import time
import pathlib
import numpy as np
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
STORE = ROOT / "2026-05-27_09-07-52.wavpack-bps2.25.zarr"
SORT_ROOT = ROOT / "sortings"
out = SORT_ROOT / "wavpack-bps2.25"

print(f"=== [{time.strftime('%T')}] sorting wavpack-bps2.25 (sort_n_jobs=3) ===", flush=True)
t0 = time.perf_counter()
agg = sort_store(STORE, out, scheme="3", scheme3_block_duration_sec=1800,
                 materialize_n_jobs=48, sort_n_jobs=3)
secs = time.perf_counter() - t0
agg.save(folder=str(out / "aggregated"), overwrite=True)
groups = np.asarray(agg.get_property("group"))
per_tetrode = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
result = {"wavpack-bps2.25": {"minutes": round(secs / 60, 1),
                              "total_units": int(agg.get_num_units()),
                              "per_tetrode": per_tetrode}}
print(f"[{time.strftime('%T')}] wavpack: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
print("RESULT " + json.dumps(result), flush=True)

# merge into the summary (keep blosc entry if present)
sfile = SORT_ROOT / "sorting_summary.json"
summary = json.loads(sfile.read_text()) if sfile.exists() else {}
summary.update(result)
sfile.write_text(json.dumps(summary, indent=2))
print("ALL DONE", flush=True)

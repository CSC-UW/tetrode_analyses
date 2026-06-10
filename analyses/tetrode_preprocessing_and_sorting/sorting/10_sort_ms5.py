"""Spike sort both compression stores of the full session with MountainSort5.

For each store: bandpass -> cross-tetrode CMR (materialized once) -> sort each
tetrode (scheme 3, ms5 per-group whitening) in parallel via
run_sorter_by_property -> aggregate. Saves the aggregated sorting + a summary.

    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/10_sort_ms5.py
"""
import json
import time
import pathlib
import numpy as np
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
STORES = {
    "blosc-zstd": ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr",
    "wavpack-bps2.25": ROOT / "2026-05-27_09-07-52.wavpack-bps2.25.zarr",
}
SORT_ROOT = ROOT / "sortings"
SORT_ROOT.mkdir(parents=True, exist_ok=True)

summary = {}
for name, store in STORES.items():
    out = SORT_ROOT / name
    print(f"\n=== [{time.strftime('%T')}] sorting {name} ===", flush=True)
    t0 = time.perf_counter()
    agg = sort_store(store, out, scheme="3", scheme3_block_duration_sec=1800,
                     materialize_n_jobs=48, sort_n_jobs=6)
    secs = time.perf_counter() - t0
    agg.save(folder=str(out / "aggregated"), overwrite=True)
    groups = np.asarray(agg.get_property("group"))
    per_tetrode = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
    summary[name] = {"minutes": round(secs / 60, 1),
                     "total_units": int(agg.get_num_units()),
                     "per_tetrode": per_tetrode}
    print(f"[{time.strftime('%T')}] {name}: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
    print("RESULT " + json.dumps({name: summary[name]}), flush=True)
    (SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))

print("ALL DONE", flush=True)

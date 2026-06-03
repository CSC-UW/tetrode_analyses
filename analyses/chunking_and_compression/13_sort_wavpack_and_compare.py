"""Wavpack-only seeded re-sort + lossless-vs-lossy comparison (resume after disk abort).

The seeded both-stores run (12_sort_seeded_and_compare.py) completed blosc-zstd
(785 units, sortings_seed42/blosc-zstd/aggregated saved) but its wavpack
materialize was killed when a co-tenant job (driessen2/slap_process) filled the
shared /nvme. This script finishes the job: it REUSES the saved blosc sorting,
sorts only wavpack-bps2.25 (same seed/params), then runs the comparison.

Disk guard: before the expensive wavpack materialize it checks /nvme free space
and aborts cleanly (no partial cache) if below DISK_FLOOR_GB, so we never drive
the shared disk toward 0 again.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SEED = 42
SORT_ROOT = ROOT / "sortings_seed42"
WAVPACK_STORE = ROOT / "2026-05-27_09-07-52.wavpack-bps2.25.zarr"
DISK_FLOOR_GB = 1800  # need ~1.3TB cmr_cache + ~0.3TB sort scratch + margin

def free_gb(path="/nvme"):
    s = shutil.disk_usage(path)
    return s.free / 1e9

# ---- disk guard ----
fg = free_gb()
print(f"[{time.strftime('%T')}] /nvme free: {fg:.0f} GB (floor {DISK_FLOOR_GB})", flush=True)
if fg < DISK_FLOOR_GB:
    print(f"ABORT: insufficient disk ({fg:.0f} GB < {DISK_FLOOR_GB} GB). "
          f"Co-tenant job likely active; retry when space frees.", flush=True)
    raise SystemExit(2)

# ---- sort wavpack only ----
out = SORT_ROOT / "wavpack-bps2.25"
print(f"\n=== [{time.strftime('%T')}] sorting wavpack-bps2.25 (seed={SEED}) ===", flush=True)
t0 = time.perf_counter()
agg_w = sort_store(WAVPACK_STORE, out, scheme="3", cmr="global", scheme3_block_duration_sec=3600,
                   whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)
secs = time.perf_counter() - t0
agg_w = agg_w.rename_units(np.arange(agg_w.get_num_units(), dtype="int64"))
agg_w.save(folder=str(out / "aggregated"), overwrite=True)
groups = np.asarray(agg_w.get_property("group"))
summary = json.loads((SORT_ROOT / "sorting_summary.json").read_text())  # has blosc already
summary["wavpack-bps2.25"] = {"minutes": round(secs / 60, 1), "total_units": int(agg_w.get_num_units()),
                              "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
(SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))
print(f"[{time.strftime('%T')}] wavpack-bps2.25: {agg_w.get_num_units()} units in {secs/60:.1f} min", flush=True)
print("RESULT " + json.dumps({"wavpack-bps2.25": summary["wavpack-bps2.25"]}), flush=True)

# ---- comparison: load blosc from disk, compare to wavpack ----
sb = si.load(str(SORT_ROOT / "blosc-zstd" / "aggregated"))
sw = agg_w
dur_s = 5215033052 / 30000.0
def fr(s): return {u: s.get_unit_spike_train(u).size / dur_s for u in s.unit_ids}
frb, frw = fr(sb), fr(sw)
def compare(thr):
    kb = [u for u, r in frb.items() if r >= thr]; kw = [u for u, r in frw.items() if r >= thr]
    sbf = sb.select_units(kb) if thr else sb; swf = sw.select_units(kw) if thr else sw
    cmp = sc.compare_two_sorters(sbf, swf, sorting1_name="blosc", sorting2_name="wavpack", match_score=0.5)
    m = cmp.get_matching()[0]; nm = int((m.values != -1).sum())
    ag = cmp.agreement_scores
    mean_ag = float(np.nanmean([ag.loc[u1, u2] for u1, u2 in m.items() if u2 != -1])) if nm else float("nan")
    return {"blosc": len(sbf.unit_ids), "wavpack": len(swf.unit_ids), "matched": nm,
            "mean_agreement": round(mean_ag, 4)}
comp = {f"FR>={thr}Hz" if thr else "all_units": compare(thr) for thr in [0.0, 0.1, 0.5, 1.0]}
comp["note"] = ("Seeded (whitening_seed=42) blosc-vs-wavpack; determinism ceiling ~0.996. "
                "Agreement below ~0.996 reflects the lossy (bps=2.25) compression effect.")
print("\nCOMPARISON " + json.dumps(comp, indent=2), flush=True)
(SORT_ROOT / "comparison_summary.json").write_text(json.dumps(comp, indent=2))
print("ALL DONE", flush=True)

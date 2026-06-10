"""Seeded re-sort of both stores + clean lossless-vs-lossy comparison.

Fixes the only meaningful non-determinism source (whitening's random chunk
selection) by passing whitening_seed to MountainSort5, so blosc-vs-wavpack
agreement measures the lossy-compression effect against the ~0.996 seeded
determinism ceiling (residual is unseedable FP/threading noise). Outputs to a
separate sortings_seed42/ dir (originals preserved).
"""
import json
import time
import pathlib
import numpy as np
import spikeinterface.comparison as sc
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SEED = 42
STORES = {
    "blosc-zstd": ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr",
    "wavpack-bps2.25": ROOT / "2026-05-27_09-07-52.wavpack-bps2.25.zarr",
}
SORT_ROOT = ROOT / "sortings_seed42"
SORT_ROOT.mkdir(parents=True, exist_ok=True)

summary = {}
aggs = {}
for name, store in STORES.items():
    out = SORT_ROOT / name
    print(f"\n=== [{time.strftime('%T')}] sorting {name} (seed={SEED}) ===", flush=True)
    t0 = time.perf_counter()
    agg = sort_store(store, out, scheme="3", cmr="global", scheme3_block_duration_sec=3600,
                     whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)
    secs = time.perf_counter() - t0
    agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
    agg.save(folder=str(out / "aggregated"), overwrite=True)
    aggs[name] = agg
    groups = np.asarray(agg.get_property("group"))
    summary[name] = {"minutes": round(secs / 60, 1), "total_units": int(agg.get_num_units()),
                     "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
    print(f"[{time.strftime('%T')}] {name}: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
    print("RESULT " + json.dumps({name: summary[name]}), flush=True)
    (SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))

# ---- clean lossless-vs-lossy comparison (all units + firing-rate filtered) ----
sb, sw = aggs["blosc-zstd"], aggs["wavpack-bps2.25"]
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

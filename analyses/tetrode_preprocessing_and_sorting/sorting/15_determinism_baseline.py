"""Full-48h determinism baseline + lossless-vs-lossy agreement (post PCA fix).

Runs three full-session MountainSort5 scheme-3 sorts with BOTH nondeterminism
sources controlled -- whitening_seed=42 (Source 1) and the deterministic-PCA
solver in SnippetClassifier/compute_pca_features (Source 2/2b):

  blosc-A, blosc-B : same lossless store, sorted twice -> DETERMINISM CEILING
                     (compare_two_sorters(A, B); residual gap is Source 3,
                      i.e. unseedable FP/threading non-associativity)
  wavpack          : lossy store -> LOSSLESS-VS-LOSSY agreement (vs blosc-A)

Only the gap between (lossless-vs-lossy) and (blosc-vs-blosc ceiling) is the
true bps=2.25 compression effect; the earlier ~0.71 number conflated it with
sorter variability that the fix now removes.

Efficiency: bandpass + global CMR + materialize is deterministic, so the blosc
store is materialized to ONE shared cache reused by blosc-A and blosc-B (the
only thing that varies between them is the sorter's own RNG -- what we measure).
A disk guard aborts cleanly before any large materialize if /nvme is tight, so
we never drive the shared disk toward 0.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import spikeinterface.comparison as sc
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SEED = 42
N_FRAMES = 5215033052          # full session, both experiments concatenated
DUR_S = N_FRAMES / 30000.0
DISK_FLOOR_GB = 1800           # ~1.3TB cmr cache + ~0.3TB sort scratch + margin
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
WAVPACK = ROOT / "2026-05-27_09-07-52.wavpack-bps2.25.zarr"
SORT_ROOT = ROOT / "sortings_seed42_pcafix"
SORT_ROOT.mkdir(parents=True, exist_ok=True)
BLOSC_CACHE = SORT_ROOT / "_blosc_cmr_cache"   # shared across blosc-A / blosc-B

SORT_KW = dict(scheme="3", cmr="global", scheme3_block_duration_sec=3600,
               whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)


def free_gb(path="/nvme"):
    return shutil.disk_usage(path).free / 1e9


def disk_guard(label):
    fg = free_gb()
    print(f"[{time.strftime('%T')}] /nvme free: {fg:.0f} GB (floor {DISK_FLOOR_GB}) before {label}", flush=True)
    if fg < DISK_FLOOR_GB:
        print(f"ABORT: insufficient disk ({fg:.0f} GB < {DISK_FLOOR_GB} GB) before {label}; "
              f"co-tenant job likely active. Retry when space frees.", flush=True)
        raise SystemExit(2)


def run_sort(store, out, *, cmr_cache_dir=None):
    out = pathlib.Path(out)
    print(f"\n=== [{time.strftime('%T')}] sorting -> {out.name} (seed={SEED}) ===", flush=True)
    t0 = time.perf_counter()
    agg = sort_store(store, out, cmr_cache_dir=cmr_cache_dir, **SORT_KW)
    secs = time.perf_counter() - t0
    agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
    agg.save(folder=str(out / "aggregated"), overwrite=True)
    groups = np.asarray(agg.get_property("group"))
    info = {"minutes": round(secs / 60, 1), "total_units": int(agg.get_num_units()),
            "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
    print(f"[{time.strftime('%T')}] {out.name}: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
    print("RESULT " + json.dumps({out.name: info}), flush=True)
    return agg, info


def fr_map(s):
    return {u: s.get_unit_spike_train(u).size / DUR_S for u in s.unit_ids}


def compare_pair(sa, sb, name_a, name_b):
    fa, fb = fr_map(sa), fr_map(sb)

    def at(thr):
        ka = [u for u, r in fa.items() if r >= thr]
        kb = [u for u, r in fb.items() if r >= thr]
        saf = sa.select_units(ka) if thr else sa
        sbf = sb.select_units(kb) if thr else sb
        cmp = sc.compare_two_sorters(saf, sbf, sorting1_name=name_a, sorting2_name=name_b, match_score=0.5)
        m = cmp.get_matching()[0]
        nm = int((m.values != -1).sum())
        ag = cmp.agreement_scores
        mean_ag = float(np.nanmean([ag.loc[u1, u2] for u1, u2 in m.items() if u2 != -1])) if nm else float("nan")
        return {name_a: len(saf.unit_ids), name_b: len(sbf.unit_ids), "matched": nm,
                "mean_agreement": round(mean_ag, 4)}

    return {(f"FR>={thr}Hz" if thr else "all_units"): at(thr) for thr in [0.0, 0.1, 0.5, 1.0]}


# ---------------------------------------------------------------- run -------
summary = {}

# blosc-A: materializes the shared blosc cache
disk_guard("blosc-A materialize")
agg_a, summary["blosc-A"] = run_sort(BLOSC, SORT_ROOT / "blosc-A", cmr_cache_dir=BLOSC_CACHE)
(SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))

# blosc-B: reuses the shared blosc cache (no re-materialize)
agg_b, summary["blosc-B"] = run_sort(BLOSC, SORT_ROOT / "blosc-B", cmr_cache_dir=BLOSC_CACHE)
(SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))

# done with the shared blosc cache
shutil.rmtree(BLOSC_CACHE, ignore_errors=True)
print(f"[{time.strftime('%T')}] removed shared blosc cmr cache", flush=True)

# wavpack: own cache (auto-deleted by sort_store)
disk_guard("wavpack materialize")
agg_w, summary["wavpack-bps2.25"] = run_sort(WAVPACK, SORT_ROOT / "wavpack-bps2.25")
(SORT_ROOT / "sorting_summary.json").write_text(json.dumps(summary, indent=2))

# ---------------------------------------------------------- comparisons -----
ceiling = compare_pair(agg_a, agg_b, "blosc_A", "blosc_B")
ceiling["note"] = ("DETERMINISM CEILING: same lossless store sorted twice with "
                   "whitening_seed=42 + deterministic-PCA solver. Residual gap below 1.0 is "
                   "unseedable FP/threading non-associativity (Source 3).")
lossy = compare_pair(agg_a, agg_w, "blosc", "wavpack")
lossy["note"] = ("LOSSLESS-VS-LOSSY: blosc-A vs wavpack (bps=2.25), same seed+fix. "
                 "Compare against the blosc-vs-blosc ceiling above; only the shortfall "
                 "relative to the ceiling is the true lossy-compression effect.")
comp = {"determinism_ceiling_blosc_vs_blosc": ceiling, "lossless_vs_lossy_blosc_vs_wavpack": lossy}
print("\nCOMPARISON " + json.dumps(comp, indent=2), flush=True)
(SORT_ROOT / "comparison_summary.json").write_text(json.dumps(comp, indent=2))
print("ALL DONE", flush=True)

"""Full-48h sorting agreement for WavPack bps=6.0 vs the lossless reference.

bps=2.25 sorting agreement (~0.71) was judged too low for production; this tests
a much higher-fidelity lossy setting (bps=6.0: in-band err 0.18 uV = 0.01x the
noise floor, ratio 2.66x ~251 GB; see README bps table) to see whether sorting
agreement recovers.

Pipeline (identical to the canonical blosc-A / wavpack runs, so the comparison is
apples-to-apples): global CMR, materialize float32, MountainSort5 scheme 3,
block 3600 s, whitening_seed=42, deterministic PCA. Reference = the deterministic
lossless float32 sort blosc-A (781 units; determinism ceiling vs blosc-B = 1.0),
so any shortfall below 1.0 is the bps=6.0 lossy-compression effect.

The bps=6.0 store is created from the lossless blosc store (bit-exact to raw,
local, no NFS) via the same convert_recording used for the bps=2.25 store.
"""
import json
import shutil
import time
import pathlib
import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
from tetrode_analyses.spikeinterface import convert_recording
from tetrode_analyses.sorting import sort_store

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SEED = 42
BPS = 6.0
N_FRAMES = 5215033052
DUR_S = N_FRAMES / 30000.0
DISK_FLOOR_GB = 1800
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
BPS6_STORE = ROOT / "2026-05-27_09-07-52.wavpack-bps6.0.zarr"
SORT_ROOT = ROOT / "sortings_seed42_pcafix"
REF_AGG = SORT_ROOT / "blosc-A" / "aggregated"      # deterministic float32 reference
OUT = SORT_ROOT / "wavpack-bps6.0"

SORT_KW = dict(scheme="3", cmr="global", scheme3_block_duration_sec=3600,
               whitening_seed=SEED, materialize_n_jobs=96, sort_n_jobs=5)


def free_gb(path="/nvme"):
    return shutil.disk_usage(path).free / 1e9


def disk_guard(label):
    fg = free_gb()
    print(f"[{time.strftime('%T')}] /nvme free: {fg:.0f} GB (floor {DISK_FLOOR_GB}) before {label}", flush=True)
    if fg < DISK_FLOOR_GB:
        print(f"ABORT: insufficient disk ({fg:.0f} GB < {DISK_FLOOR_GB} GB) before {label}.", flush=True)
        raise SystemExit(2)


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


# ---- create the bps=6.0 store from the lossless blosc store (if needed) ----
if not BPS6_STORE.exists():
    disk_guard("bps6.0 store creation")
    print(f"\n=== [{time.strftime('%T')}] creating {BPS6_STORE.name} from blosc (bit-exact source) ===", flush=True)
    t0 = time.perf_counter()
    rec = si.read_zarr(str(BLOSC))
    convert_recording(rec, BPS6_STORE, compressor="wavpack", bps=BPS, n_jobs=32)
    disk = sum(p.stat().st_size for p in BPS6_STORE.rglob("*") if p.is_file())
    print(f"[{time.strftime('%T')}] bps6.0 store: {disk/1e9:.1f} GB, ratio {N_FRAMES*64*2/disk:.2f}x "
          f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
else:
    print(f"[{time.strftime('%T')}] reusing existing {BPS6_STORE.name}", flush=True)

# ---- sort the bps=6.0 store (same pipeline as blosc-A) ----
disk_guard("bps6.0 sort")
print(f"\n=== [{time.strftime('%T')}] sorting {BPS6_STORE.name} (seed={SEED}) ===", flush=True)
t0 = time.perf_counter()
agg = sort_store(BPS6_STORE, OUT, **SORT_KW)
secs = time.perf_counter() - t0
agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
agg.save(folder=str(OUT / "aggregated"), overwrite=True)
groups = np.asarray(agg.get_property("group"))
info = {"minutes": round(secs / 60, 1), "total_units": int(agg.get_num_units()),
        "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
print(f"[{time.strftime('%T')}] wavpack-bps6.0: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
print("RESULT " + json.dumps({"wavpack-bps6.0": info}), flush=True)

ss_path = SORT_ROOT / "sorting_summary.json"
ss = json.loads(ss_path.read_text()) if ss_path.exists() else {}
ss["wavpack-bps6.0"] = info
ss_path.write_text(json.dumps(ss, indent=2))

# ---- compare vs deterministic lossless float32 reference ----
ref = si.load(str(REF_AGG))
comp = {"float32_lossless_vs_bps6.0": compare_pair(ref, agg, "blosc", "wavpack_bps6")}
comp["float32_lossless_vs_bps6.0"]["note"] = (
    "bps=6.0 lossy-compression effect: lossless float32 reference (blosc-A, deterministic) "
    "vs WavPack bps=6.0, same seed + deterministic PCA. Determinism ceiling = 1.0, so any "
    "shortfall is purely the bps=6.0 compression. Compare to bps=2.25 (~0.71) and int16 (~0.78).")
print("\nCOMPARISON " + json.dumps(comp, indent=2), flush=True)
(SORT_ROOT / "comparison_bps6_summary.json").write_text(json.dumps(comp, indent=2))
print("ALL DONE", flush=True)

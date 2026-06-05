"""Full-48h int16-quantization test: blosc sorted with an int16 materialized
binary vs the float32 reference.

The sorter is now fully deterministic (whitening_seed=42 + deterministic-PCA;
blosc-vs-blosc = 1.0 at 48 h, see 15_determinism_baseline.py), so the ONLY
difference between this sort and the float32 reference is the dtype of the
materialized bandpass+global-CMR binary:

  reference : sortings_seed42_pcafix/blosc-A/aggregated  (float32, already on disk)
  this run  : sortings_seed42_pcafix/blosc-int16          (materialize_dtype=int16)

bandpass + CMR are computed in float32 and then written as int16, i.e. the
preprocessed sorter input is quantized to integer ADC-count resolution (the raw
acquisition resolution; gain 0.195 uV/count not applied at this stage). On this
data post-CMR std ~80 counts, peaks ~1-2k, no int16 clipping, ~48 dB quant SNR.
Any agreement < 1.0 vs the float32 reference is the pure int16-quantization
effect on the sort.
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
N_FRAMES = 5215033052
DUR_S = N_FRAMES / 30000.0
DISK_FLOOR_GB = 1500           # int16 cache ~650 GB + sort scratch + margin
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SORT_ROOT = ROOT / "sortings_seed42_pcafix"
REF_AGG = SORT_ROOT / "blosc-A" / "aggregated"   # float32 reference (deterministic)
OUT = SORT_ROOT / "blosc-int16"

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
disk_guard("blosc-int16 materialize")
print(f"\n=== [{time.strftime('%T')}] sorting blosc -> int16 binary (seed={SEED}) ===", flush=True)
t0 = time.perf_counter()
agg = sort_store(BLOSC, OUT, materialize_dtype="int16", **SORT_KW)
secs = time.perf_counter() - t0
agg = agg.rename_units(np.arange(agg.get_num_units(), dtype="int64"))
agg.save(folder=str(OUT / "aggregated"), overwrite=True)
groups = np.asarray(agg.get_property("group"))
info = {"minutes": round(secs / 60, 1), "total_units": int(agg.get_num_units()),
        "per_tetrode": {int(g): int((groups == g).sum()) for g in np.unique(groups)}}
print(f"[{time.strftime('%T')}] blosc-int16: {agg.get_num_units()} units in {secs/60:.1f} min", flush=True)
print("RESULT " + json.dumps({"blosc-int16": info}), flush=True)

# update sorting_summary.json in place
ss_path = SORT_ROOT / "sorting_summary.json"
ss = json.loads(ss_path.read_text()) if ss_path.exists() else {}
ss["blosc-int16"] = info
ss_path.write_text(json.dumps(ss, indent=2))

# ---------------------------------------------------------- comparison ------
ref = si.load(str(REF_AGG))   # float32 blosc-A
comp = {"float32_vs_int16": compare_pair(ref, agg, "blosc_float32", "blosc_int16")}
comp["float32_vs_int16"]["note"] = (
    "INT16 QUANTIZATION effect: float32 reference (blosc-A) vs int16-materialized "
    "binary, same seed + deterministic PCA. Determinism ceiling is 1.0 (see "
    "15_determinism_baseline.py), so any shortfall below 1.0 is purely the int16 "
    "(integer ADC-count) quantization of the preprocessed sorter input.")
print("\nCOMPARISON " + json.dumps(comp, indent=2), flush=True)
(SORT_ROOT / "comparison_int16_summary.json").write_text(json.dumps(comp, indent=2))
print("ALL DONE", flush=True)

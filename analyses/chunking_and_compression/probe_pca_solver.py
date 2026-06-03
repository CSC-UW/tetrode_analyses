"""Instrumented probe: measure the actual (L, n_features, solver) distribution in
SnippetClassifier.fit() during a real sort, to answer which PCA solver fires.

SnippetClassifier.fit() is instrumented (temporarily) to append a CSV row per fit
when MS5_SOLVER_LOG is set; env vars propagate to joblib workers (unlike
monkeypatches). We sort the blosc store on two durations:
  - 600 s  (matches the determinism slice test; one scheme-3 block)
  - 3600 s (the production block_duration; training still capped at 300 s)
Both train PCA on the first 300 s of the block, so L should be comparable -> the
slice is representative of every full-run block.
"""
import os
import csv
import pathlib
import collections
import numpy as np

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
STORE = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
PROBE = pathlib.Path("/nvme/neuropixels/tmp/cc_bench/solver_probe")
PROBE.mkdir(parents=True, exist_ok=True)


def summarize(csv_path, label):
    rows = []
    with open(csv_path) as f:
        for r in csv.reader(f):
            if len(r) == 7:
                rows.append(r)
    if not rows:
        print(f"\n[{label}] no fit() rows logged"); return
    Ls = np.array([int(r[0]) for r in rows])
    nfeat = np.array([int(r[3]) for r in rows])
    solvers = [r[6] for r in rows]
    by_solver = collections.Counter(solvers)
    print(f"\n=== [{label}] {len(rows)} classifier fits ===")
    print(f"  n_features (T*M): {sorted(set(nfeat.tolist()))}")
    print(f"  L (training-set size): min={Ls.min()} median={int(np.median(Ls))} mean={Ls.mean():.0f} max={Ls.max()}")
    print(f"  L percentiles: p10={int(np.percentile(Ls,10))} p25={int(np.percentile(Ls,25))} "
          f"p50={int(np.percentile(Ls,50))} p75={int(np.percentile(Ls,75))} p90={int(np.percentile(Ls,90))}")
    print(f"  solver counts: {dict(by_solver)}")
    for s in sorted(by_solver):
        m = np.array([x == s for x in solvers])
        print(f"    {s:>16s}: {by_solver[s]:4d} fits  L range [{Ls[m].min()}, {Ls[m].max()}]")
    rand_frac = by_solver.get("randomized", 0) / len(rows)
    print(f"  >>> randomized (stochastic) fraction: {rand_frac:.1%}")


def run(duration_s):
    from tetrode_analyses.sorting import sort_store
    log = PROBE / f"solver_{int(duration_s)}s.csv"
    if log.exists():
        log.unlink()
    os.environ["MS5_SOLVER_LOG"] = str(log)
    out = PROBE / f"sort_{int(duration_s)}s"
    print(f"\n##### sorting blosc-zstd first {duration_s} s (logging -> {log.name}) #####", flush=True)
    sort_store(STORE, out, scheme="3", cmr="global", scheme3_block_duration_sec=3600,
               whitening_seed=42, test_duration_s=duration_s,
               materialize_n_jobs=16, sort_n_jobs=5)
    os.environ.pop("MS5_SOLVER_LOG", None)
    return log


for dur in [600, 3600]:
    log = run(dur)
    summarize(log, f"{dur}s slice")
print("\nPROBE DONE", flush=True)

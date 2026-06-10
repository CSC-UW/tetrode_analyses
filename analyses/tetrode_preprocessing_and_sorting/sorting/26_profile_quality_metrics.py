"""Profile SpikeInterface `quality_metrics` on the 48 h tetrode analyzer (measurement-first).

The full default 19-metric set takes ~58 min single-threaded on this analyzer (232 units,
81.3 M spikes, sparse-by-tetrode = 4 ch/unit) with no progress output. This harness MEASURES
where the wall-clock goes so we know what to *optimize* (we always compute the full set; metric
parameter values are fixed by science, not tuned for speed). It never modifies the analyzer:
every compute uses save=False, and a zarr fingerprint is asserted unchanged.

Key SI mechanics used (verified against the editable checkout):
  * `analyzer.compute("quality_metrics", metric_names=[...], save=False, delete_existing_metrics=True,
    seed=SEED)` returns the *extension*; per-metric wall-times live at `ext.data["runtime_s"]`
    (BaseMetricExtension._compute_metrics times each metric individually).
  * The shared PCA `_prepare_data` step runs BEFORE the per-metric timing loop, so its cost is
    attributed to NO metric and is invisible in runtime_s. We recover it via a `[mahalanobis]`-only
    call: prep ~= total_wall - runtime_s["mahalanobis"] (mahalanobis math itself is cheap).
  * Only `nn_advanced` (excluded from the default set) honors n_jobs; the default 19 are single
    threaded. PCA-metric neighbors are same-tetrode (sparsity), so the scaling ladder subsets by
    WHOLE tetrodes to keep each unit's neighbor set intact.

Stages (opt-in via --stages; default `preflight` is fast & safe):
  preflight : load, assert extensions + zarr-unchanged contract, print structure. ~1 min.
  scan      : tetrode-count ladder {1,2,4,8,16} -> per-metric runtime_s + prep + overhead at each N.
              Top rung (232 units) IS the full-N breakdown. Writes per_metric + scaling CSV/JSON/PNG.
              ~2 h for the full ladder (dominated by the 232 rung).
  cprofile  : cProfile a single metric at full N (use after `scan` identifies the long pole).
  params    : sweep cost-relevant params of a metric to see what drives its cost (diagnostic only).

Usage:
  cd gfys_workspace
  uv run --all-extras --group dev python .../26_profile_quality_metrics.py --stages preflight
  uv run --all-extras --group dev python .../26_profile_quality_metrics.py --stages scan --ladder 1,2
  uv run --all-extras --group dev python .../26_profile_quality_metrics.py --stages scan          # full
  uv run --all-extras --group dev python .../26_profile_quality_metrics.py --stages cprofile --metric drift
  uv run --all-extras --group dev python .../26_profile_quality_metrics.py --stages params --metric sliding_rp_violation
"""
import argparse
import cProfile
import json
import os
import pstats
import threading
import time
import pathlib

import numpy as np
import psutil
import spikeinterface.full as si

ANALYZER = pathlib.Path(
    "/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/"
    "sortings_seed42_pcafix/blosc-43200s-train3600s/analyzer.zarr"
)
OUTDIR = pathlib.Path(__file__).resolve().parent
SEED = 42

# The SI default set (nn_advanced excluded by SI as "too slow"). We always compute ALL of these.
DEFAULT_19 = [
    "num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation", "rp_violation",
    "sliding_rp_violation", "synchrony", "firing_range", "amplitude_cv", "amplitude_cutoff",
    "noise_cutoff", "amplitude_median", "drift", "sd_ratio", "mahalanobis", "d_prime",
    "nearest_neighbor", "silhouette",
]
PCA_METRICS = ["mahalanobis", "d_prime", "nearest_neighbor", "silhouette"]
# Cheapest PCA metric, used to isolate the shared _prepare_data cost.
PREP_PROBE = "mahalanobis"

# Parameter cost-driver grids for the `params` stage (diagnostic; defaults listed first).
PARAM_GRIDS = {
    "sliding_rp_violation": [("bin_size_ms", v) for v in (0.25, 0.5, 1.0)],
    "drift": [("interval_s", v) for v in (60, 120, 300)],
    "silhouette": [("method", v) for v in ("simplified", "full")],
    "nearest_neighbor": [("max_spikes", v) for v in (10000, 2000, 1000)],
}


class TreeSampler:
    """Peak USS (private working set; excludes mmap'd file cache) + RSS + CPU over the process tree.
    Copied from 23_bench_training_duration.py."""

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
        self.peak_uss = self.peak_rss = 0
        self.peak_cpu = 0.0
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


def log(msg):
    print(f"[{time.strftime('%T')}] {msg}", flush=True)


def zarr_fingerprint(root):
    """Read-only fingerprint of on-disk state: (max mtime, file count, total bytes)."""
    mx, n, tot = 0.0, 0, 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                st = os.stat(os.path.join(dirpath, f))
            except OSError:
                continue
            mx = max(mx, st.st_mtime)
            n += 1
            tot += st.st_size
    return (round(mx, 3), n, tot)


def time_metrics(analyzer, metric_names, sampler=None):
    """Compute `metric_names` with save=False (delete_existing -> clean isolation). Returns dict with
    total wall, per-metric runtime_s, and (if sampler given) peak USS GB / CPU cores."""
    if sampler is not None:
        sampler.start()
    t0 = time.perf_counter()
    ext = analyzer.compute(
        "quality_metrics", metric_names=metric_names, save=False,
        delete_existing_metrics=True, seed=SEED,
    )
    total = time.perf_counter() - t0
    if sampler is not None:
        sampler.stop()
    out = {"total": total, "runtime_s": dict(ext.data.get("runtime_s", {}))}
    if sampler is not None:
        out["peak_uss_gb"] = round(sampler.peak_uss / 1e9, 2)
        out["peak_cpu_cores"] = round(sampler.peak_cpu / 100, 2)
    return out


def tetrode_groups(analyzer):
    """Ordered list of (group_id, [unit_ids]) using the sorting 'group' property."""
    groups = analyzer.sorting.get_property("group")
    uids = np.asarray(analyzer.unit_ids)
    out = []
    for g in sorted(np.unique(groups).tolist()):
        out.append((g, uids[groups == g].tolist()))
    return out


# ----------------------------------------------------------------------------- stages


def stage_preflight(analyzer):
    log(f"format={analyzer.format} n_units={analyzer.get_num_units()} sparse={analyzer.is_sparse()} "
        f"fs={analyzer.sampling_frequency} dur_h={analyzer.get_total_duration()/3600:.2f}")
    grps = tetrode_groups(analyzer)
    sizes = [len(u) for _, u in grps]
    log(f"tetrodes={len(grps)} units/group min/med/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    pc = analyzer.get_extension("principal_components")
    log(f"pca params={pc.params}")
    for e in ["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes",
              "principal_components", "spike_locations"]:
        assert analyzer.has_extension(e), f"missing required extension {e}"
    log("all required extensions present")
    fp0 = zarr_fingerprint(ANALYZER)
    sub = analyzer.select_units(list(analyzer.unit_ids[:8]), format="memory")
    res = time_metrics(sub, ["num_spikes", "firing_rate", "snr", "mahalanobis"])
    fp1 = zarr_fingerprint(ANALYZER)
    log(f"cheap 8-unit probe: total={res['total']:.3f}s runtime_s={ {k: round(v,4) for k,v in res['runtime_s'].items()} }")
    assert fp0 == fp1, f"ZARR CHANGED by save=False compute! {fp0} != {fp1}"
    log("zarr fingerprint UNCHANGED after save=False compute -> read-only contract holds")


def measure_at_n(analyzer_n, n_units, n_tetrodes, sampler):
    """Full-set per-metric breakdown + prep + overhead at one ladder rung."""
    log(f"  full set ({len(DEFAULT_19)} metrics) on N={n_units} units ...")
    full = time_metrics(analyzer_n, DEFAULT_19, sampler=sampler)
    log(f"  full set total={full['total']:.1f}s peak_uss={full['peak_uss_gb']}GB cpu={full['peak_cpu_cores']} cores")
    log(f"  prep probe [{PREP_PROBE}] (isolates _prepare_data) ...")
    probe = time_metrics(analyzer_n, [PREP_PROBE])
    prep = probe["total"] - probe["runtime_s"].get(PREP_PROBE, 0.0)
    rt = full["runtime_s"]
    overhead = full["total"] - sum(rt.values()) - prep
    rec = {
        "n_tetrodes": n_tetrodes, "n_units": n_units,
        "total_s": round(full["total"], 3),
        "prep_s": round(prep, 3),
        "overhead_s": round(overhead, 3),
        "peak_uss_gb": full["peak_uss_gb"], "peak_cpu_cores": full["peak_cpu_cores"],
        "runtime_s": {k: round(v, 4) for k, v in rt.items()},
    }
    return rec


def stage_scan(analyzer, ladder):
    grps = tetrode_groups(analyzer)  # ordered whole tetrodes
    fp0 = zarr_fingerprint(ANALYZER)
    sampler = TreeSampler(interval=2.0)
    records = []
    for k in ladder:
        k = min(k, len(grps))
        unit_ids = [u for _, units in grps[:k] for u in units]
        n = len(unit_ids)
        log(f"=== ladder rung: {k} tetrodes, {n} units ===")
        if k == len(grps):
            an_n = analyzer  # full set: use the analyzer directly (avoid select_units overhead)
        else:
            t0 = time.perf_counter()
            an_n = analyzer.select_units(unit_ids, format="memory")
            log(f"  select_units({n}) took {time.perf_counter()-t0:.1f}s")
        rec = measure_at_n(an_n, n, k, sampler)
        records.append(rec)
        log("  RUNG " + json.dumps({kk: rec[kk] for kk in ("n_tetrodes", "n_units", "total_s", "prep_s", "overhead_s")}))
        _write_scan_outputs(records, grps)  # incremental save after every rung
    fp1 = zarr_fingerprint(ANALYZER)
    assert fp0 == fp1, f"ZARR CHANGED during scan! {fp0} != {fp1}"
    log("zarr fingerprint UNCHANGED across full scan")
    _report_breakdown(records[-1])


def _write_scan_outputs(records, grps):
    import csv
    # long-format scaling CSV: one row per (n_units, metric|__prep__|__overhead__|__total__)
    scaling_csv = OUTDIR / "profile_quality_metrics_scaling.csv"
    with open(scaling_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_tetrodes", "n_units", "key", "seconds"])
        for r in records:
            for m, s in r["runtime_s"].items():
                w.writerow([r["n_tetrodes"], r["n_units"], m, s])
            w.writerow([r["n_tetrodes"], r["n_units"], "__prep__", r["prep_s"]])
            w.writerow([r["n_tetrodes"], r["n_units"], "__overhead__", r["overhead_s"]])
            w.writerow([r["n_tetrodes"], r["n_units"], "__total__", r["total_s"]])
    (OUTDIR / "profile_quality_metrics_scan.json").write_text(json.dumps(records, indent=2))

    # per-metric ranked breakdown at the largest rung computed so far
    top = records[-1]
    rows = [(m, s, 100 * s / top["total_s"]) for m, s in top["runtime_s"].items()]
    rows.append(("__pca_prepare_data__", top["prep_s"], 100 * top["prep_s"] / top["total_s"]))
    rows.append(("__overhead__", top["overhead_s"], 100 * top["overhead_s"] / top["total_s"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    pm_csv = OUTDIR / "profile_quality_metrics_per_metric.csv"
    with open(pm_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "seconds", "pct_of_total", "n_units"])
        for m, s, pct in rows:
            w.writerow([m, round(s, 3), round(pct, 2), top["n_units"]])
    (OUTDIR / "profile_quality_metrics_per_metric.json").write_text(json.dumps(
        {"n_units": top["n_units"], "total_s": top["total_s"], "ranked": rows}, indent=2))


def _report_breakdown(top):
    rows = [(m, s) for m, s in top["runtime_s"].items()]
    rows.append(("__pca_prepare_data__", top["prep_s"]))
    rows.append(("__overhead__", top["overhead_s"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"\n--- QUALITY_METRICS PER-METRIC BREAKDOWN (N={top['n_units']} units, total={top['total_s']:.1f}s) ---", flush=True)
    print(f"{'metric':>24}{'seconds':>12}{'pct':>8}", flush=True)
    for m, s in rows:
        print(f"{m:>24}{s:>12.2f}{100*s/top['total_s']:>7.1f}%", flush=True)
    # try to plot scaling
    try:
        _plot_scaling()
    except Exception as e:  # plotting is best-effort
        log(f"(scaling plot skipped: {e})")


def _plot_scaling():
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    series = defaultdict(list)  # key -> list[(n_units, seconds)]
    with open(OUTDIR / "profile_quality_metrics_scaling.csv") as fh:
        for row in csv.DictReader(fh):
            series[row["key"]].append((int(row["n_units"]), float(row["seconds"])))
    if not series or max(len(v) for v in series.values()) < 2:
        return  # need >=2 rungs to show a curve
    # plot the heaviest keys at the top rung
    top_n = max(n for v in series.values() for n, _ in v)
    heaviest = sorted(series, key=lambda k: dict(series[k]).get(top_n, 0), reverse=True)
    keep = [k for k in heaviest if k != "__total__"][:8] + ["__total__"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in keep:
        pts = sorted(series[k])
        xs = [p[0] for p in pts]
        ys = [max(p[1], 1e-6) for p in pts]
        # log-log slope from first to last
        slope = np.polyfit(np.log(xs), np.log(ys), 1)[0] if len(xs) >= 2 else float("nan")
        ax.plot(xs, ys, marker="o", label=f"{k} (slope~{slope:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_units (whole tetrodes)"); ax.set_ylabel("seconds")
    ax.set_title("quality_metrics cost vs n_units")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "profile_quality_metrics_scaling.png", dpi=120)
    log(f"wrote {OUTDIR/'profile_quality_metrics_scaling.png'}")


def stage_cprofile(analyzer, metric):
    log(f"cProfile of [{metric}] at N={analyzer.get_num_units()} (cumulative time, top 30)")
    pr = cProfile.Profile()
    pr.enable()
    analyzer.compute("quality_metrics", metric_names=[metric], save=False,
                     delete_existing_metrics=True, seed=SEED)
    pr.disable()
    out = OUTDIR / f"profile_{metric}_cprofile.txt"
    with open(out, "w") as fh:
        st = pstats.Stats(pr, stream=fh).sort_stats("cumulative")
        st.print_stats(30)
    log(f"wrote {out}")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


def stage_params(analyzer, metric):
    grid = PARAM_GRIDS.get(metric)
    if not grid:
        log(f"no param grid defined for {metric}; known: {list(PARAM_GRIDS)}")
        return
    log(f"param cost-driver sweep for [{metric}] at N={analyzer.get_num_units()} (diagnostic only)")
    rows = []
    for pname, pval in grid:
        # the grid lists the default value first, so the default cost is measured naturally
        t0 = time.perf_counter()
        ext = analyzer.compute("quality_metrics", metric_names=[metric], save=False,
                               delete_existing_metrics=True, seed=SEED,
                               metric_params={metric: {pname: pval}})
        dt = time.perf_counter() - t0
        rt = ext.data.get("runtime_s", {}).get(metric, dt)
        rows.append({"metric": metric, "param": pname, "value": pval, "metric_s": round(rt, 3), "call_s": round(dt, 3)})
        log(f"  {pname}={pval}: metric_s={rt:.2f}")
    import csv
    out = OUTDIR / "profile_quality_metrics_param_cost.csv"
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "param", "value", "metric_s", "call_s"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    log(f"appended {len(rows)} rows to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="preflight",
                    help="comma list: preflight,scan,cprofile,params")
    ap.add_argument("--ladder", default="1,2,4,8,16",
                    help="tetrode-count rungs for scan")
    ap.add_argument("--metric", default=None, help="target metric for cprofile/params")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    log(f"loading analyzer {ANALYZER}")
    t0 = time.perf_counter()
    an = si.load_sorting_analyzer(str(ANALYZER))
    log(f"loaded in {time.perf_counter()-t0:.1f}s")

    if "preflight" in stages:
        stage_preflight(an)
    if "scan" in stages:
        ladder = [int(x) for x in args.ladder.split(",")]
        stage_scan(an, ladder)
    if "cprofile" in stages:
        assert args.metric, "--metric required for cprofile"
        stage_cprofile(an, args.metric)
    if "params" in stages:
        assert args.metric, "--metric required for params"
        stage_params(an, args.metric)
    log("DONE")


if __name__ == "__main__":
    main()

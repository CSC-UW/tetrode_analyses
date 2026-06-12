"""Operating-point sweep for chunk+track on a drift-stable epoch.

Sweeps chunk size x overlap (each requires re-sorting -> the expensive axis) and,
for free, the chaining thresholds (jaccard_min x cosine_min) by re-running
chain_matches over the cached edges from each track_span. Scores every config
against the single-sort reference (as in 31) and writes a table to pick the
operating point. Gated: run only after 31 shows the approach reconstructs the
single sort at one config.

The cosine gate is also probed here (risk #3): for the chosen (chunk_s, overlap),
chaining is re-scored with cosine_min=-1 (off) vs the sweep values, to see whether
the geometry-free template cosine ever changes the accepted edges beyond Jaccard.
"""
import json
import pathlib
import shutil
import time

import spikeinterface as si

import _track_eval as ev
from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
WORK = SR / "track_eval"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0

EPOCH_S = (36000.0, 43200.0)  # 2 h drift-stable epoch (same as 31)
CHUNK_GRID = [900.0, 1800.0, 3600.0]
OVERLAP_GRID = [0.25, 0.5]
JACCARD_GRID = [0.3, 0.4, 0.5]
COSINE_GRID = [-1.0, 0.85, 0.9, 0.95]  # -1.0 == cosine gate OFF (probe risk #3)


def main():
    si.set_global_job_kwargs(n_jobs=5, progress_bar=False, chunk_duration="1s")
    WORK.mkdir(parents=True, exist_ok=True)
    t0f, t1f = int(EPOCH_S[0] * FS), int(EPOCH_S[1] * FS)

    print(f"[{time.strftime('%T')}] building single-sort reference for {EPOCH_S} ...", flush=True)
    ref_sorting, ref_an, qm_df, ref_bin = ev.single_sort_reference(BLOSC, WORK / "ref", t0_frame=t0f, t1_frame=t1f, fs=FS)
    print(f"  reference: {ref_sorting.get_num_units()} units", flush=True)

    rows = []
    for chunk_s in CHUNK_GRID:
        for overlap in OVERLAP_GRID:
            tag = f"c{int(chunk_s)}_o{overlap}"
            print(f"\n[{time.strftime('%T')}] === config {tag} === /nvme free "
                  f"{shutil.disk_usage('/nvme').free/1e9:.0f} GB", flush=True)
            t0 = time.perf_counter()
            res = tk.track_span(
                BLOSC, WORK / f"sweep_{tag}",
                chunk_s=chunk_s, overlap_frac=overlap, t0_frame=t0f, t1_frame=t1f,
                jaccard_min=0.5, cosine_min=0.9,  # placeholder; thresholds swept below
            )
            span_min = (time.perf_counter() - t0) / 60
            n_chunks = len(res["chunks"])
            # free threshold sub-sweep over cached edges
            for jmin in JACCARD_GRID:
                for cmin in COSINE_GRID:
                    n2g, prov = tk.chain_matches(res["edges"], jaccard_min=jmin, cosine_min=cmin)
                    gs, _ = tk.assemble_global_sorting(res["chunk_sortings"], res["chunks"], n2g, fs=FS)
                    score = ev.score_reconstruction(ref_sorting, qm_df, gs)
                    mod = score["moderate"]
                    rows.append({
                        "chunk_s": chunk_s, "overlap": overlap, "n_chunks": n_chunks,
                        "jaccard_min": jmin, "cosine_min": cmin,
                        "n_global_units": gs.get_num_units(),
                        "n_candidate_edges": len(res["edges"]),
                        "all_match_frac": score["all_units"]["match_frac"],
                        "moderate_match_frac": mod["match_frac"],
                        "moderate_mean_agreement": mod["mean_agreement"],
                        "moderate_n_ref": mod["n_ref"],
                        "span_min": round(span_min, 1),
                    })
            best = max((r for r in rows if r["chunk_s"] == chunk_s and r["overlap"] == overlap),
                       key=lambda r: (r["moderate_match_frac"], r["moderate_mean_agreement"]))
            print(f"  {tag}: {n_chunks} chunks, {span_min:.1f} min; best moderate "
                  f"frac={best['moderate_match_frac']} ag={best['moderate_mean_agreement']} "
                  f"@ jmin={best['jaccard_min']} cmin={best['cosine_min']}", flush=True)
            shutil.rmtree(WORK / f"sweep_{tag}", ignore_errors=True)

    # rank configs by (moderate match_frac, mean_agreement), penalize over-fragmentation lightly
    rows.sort(key=lambda r: (r["moderate_match_frac"], r["moderate_mean_agreement"]), reverse=True)
    summary = {"epoch_s": EPOCH_S, "n_ref_units": int(ref_sorting.get_num_units()),
               "grid": {"chunk_s": CHUNK_GRID, "overlap": OVERLAP_GRID,
                        "jaccard_min": JACCARD_GRID, "cosine_min": COSINE_GRID},
               "rows": rows, "best": rows[0] if rows else None}
    (OUTDIR / "track_param_sweep.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n--- PARAM SWEEP (top 12 by moderate match_frac, mean_agreement) ---", flush=True)
    print(f"{'chunk_s':>7}{'ovl':>5}{'jmin':>5}{'cmin':>6}{'#glob':>6}{'frac':>7}{'mean_ag':>9}", flush=True)
    for r in rows[:12]:
        print(f"{r['chunk_s']:>7.0f}{r['overlap']:>5}{r['jaccard_min']:>5}{r['cosine_min']:>6}"
              f"{r['n_global_units']:>6}{r['moderate_match_frac']:>7.3f}{r['moderate_mean_agreement']:>9.4f}", flush=True)
    # cosine-gate probe: same (chunk,overlap,jmin), cosine off vs on
    print("\n--- cosine-gate probe (does cosine change accepted edges beyond Jaccard?) ---", flush=True)
    for r in rows:
        if r["cosine_min"] == -1.0:
            same = [x for x in rows if x["chunk_s"] == r["chunk_s"] and x["overlap"] == r["overlap"]
                    and x["jaccard_min"] == r["jaccard_min"] and x["cosine_min"] == 0.9]
            if same:
                s = same[0]
                if s["n_global_units"] != r["n_global_units"]:
                    print(f"  c{int(r['chunk_s'])}_o{r['overlap']} jmin={r['jaccard_min']}: "
                          f"cosine OFF #glob={r['n_global_units']} -> cmin0.9 #glob={s['n_global_units']}", flush=True)
    print("ALL DONE", flush=True)
    shutil.rmtree(WORK / "ref", ignore_errors=True)


if __name__ == "__main__":
    main()

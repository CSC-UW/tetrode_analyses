"""KS4-per-chunk tracking: does Kilosort4's deconvolution lift the MS5 isolation ceiling?

The negative result for MS5 chunk-and-match (TRACKING_FINDINGS.md): clean units break
~9%/boundary, and the breaks are NOT a matching failure (~1%) but a per-chunk ISOLATION
failure (~7%) -- MS5's local clustering simply doesn't isolate, in chunk B, a unit it
isolates in chunk A, so there's no signal to match. That caps the clean-unit
per-boundary bridge rate at ~0.91 and 0.91^95 -> 0.

Hypothesis under test: a deconvolution / template-matching sorter (KS4) detects every
spike that matches a template, so it should NOT drop a unit between adjacent chunks the
way clustering does -> a higher per-boundary bridge rate. This script sorts the SAME
overlapping chunks with KS4 (no drift correction; the geometry is fictional) and
measures the decisive number -- the clean-unit per-boundary bridge rate -- then
recomputes the IDENTICAL diagnostic for MS5 on the SAME chunk grid from the existing
checkpoints (free; no MS5 re-sort) for an airtight head-to-head.

Epoch: [36000, 57600) s -- the 6 h drift-stable, pre-discontinuity window used by
31_track_reconstruct_vs_singlesort.py. chunk_s=3600, overlap=0.5 -> 11 full chunks,
10 boundaries, all 3600 s. KS4 local chunks 0..10 align frame-for-frame with the
full-recording MS5 checkpoints 20..30 (both start at i*1800 s).

"Clean" = max member template peak-to-peak >= 170 uV (CLEAN_AMP_UV, matching
36_track_global_cluster.py). "Bridged" = a reciprocal overlap match with jaccard >=
0.3 in the next chunk.

Shares the host V100 with any co-running KS4 (each uses only a few GB of the 32 GB
card); aborts up front only if the GPU is already nearly full (near-OOM). Under sharing,
`ks4_minutes` is contended and not a valid timing benchmark -- the bridge RATE
(correctness) is what this experiment measures and is unaffected.
"""
import json
import shutil
import subprocess
import time
import pathlib
from collections import defaultdict

import numpy as np
import spikeinterface as si

from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
MS5_CHUNKS = SR / "tracked_48h" / "chunks"          # existing MS5 checkpoints (chunk_000..095)
MS5_EDGES = SR / "tracked_48h" / "edges.json"
WORK = SR / "track_ks4_epoch"                       # KS4 per-chunk work + checkpoints
OUTDIR = pathlib.Path(__file__).resolve().parent

FS = 30000.0
EPOCH_S = (36000.0, 57600.0)
CHUNK_S, OVERLAP = 3600.0, 0.5
JACCARD_MIN, AMP_THR = 0.3, 170.0
MS5_CHUNK0 = 20  # full-recording MS5 chunk index aligned with KS4 local chunk 0
# Near-OOM guard only. A co-running KS4 (e.g. another agent's whole-recording sort)
# uses only a few GB on the 32 GB V100 and shares fine; abort only if the card is
# nearly full so this run cannot fit. Timing (`ks4_minutes`) is then contended and not
# a valid benchmark, but the bridge RATE -- a correctness metric -- is unaffected.
GPU_FULL_MB = 28000


def gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0


def template_pp(t):
    return float(np.asarray(t).max() - np.asarray(t).min())


def bridge_rate(amps_by_chunk, edges, chunk_idx_list, *, amp_thr=AMP_THR, jacc_thr=JACCARD_MIN):
    """Clean-unit per-boundary bridge rate over consecutive chunks.

    ``amps_by_chunk[k]`` = {unit_a: peak-to-peak uV}. ``edges`` carry chunk_a/unit_a/
    jaccard/reciprocal. A clean unit in chunk k is 'bridged' iff it is the source of a
    reciprocal edge with jaccard >= jacc_thr into chunk k+1.
    """
    qual = defaultdict(set)
    for e in edges:
        if e["reciprocal"] and e["jaccard"] >= jacc_thr:
            qual[e["chunk_a"]].add(e["unit_a"])
    n_clean = n_bridged = n_bnd = 0
    per_boundary = []
    for k in chunk_idx_list[:-1]:
        clean = [u for u, a in amps_by_chunk.get(k, {}).items() if a >= amp_thr]
        if not clean:
            continue
        b = sum(1 for u in clean if u in qual.get(k, set()))
        per_boundary.append({"chunk_a": int(k), "n_clean": len(clean), "n_bridged": int(b)})
        n_bnd += 1
        n_clean += len(clean)
        n_bridged += b
    rate = n_bridged / n_clean if n_clean else float("nan")
    return {
        "n_boundaries": n_bnd, "n_clean": n_clean, "n_bridged": n_bridged,
        "bridge_rate": round(rate, 3) if n_clean else None,
        "implied_survival_10": round(rate ** 10, 4) if n_clean else None,
        "implied_survival_95": round(rate ** 95, 6) if n_clean else None,
        "per_boundary": per_boundary,
    }


def run_ks4_chunks(rec, groups, gci):
    """KS4-per-chunk over the epoch. Returns (amps_by_local_chunk, edges, n_chunks)."""
    t0f, t1f = int(EPOCH_S[0] * FS), int(EPOCH_S[1] * FS)
    chunks = tk.plan_chunks(rec.get_num_frames(), FS, chunk_s=CHUNK_S, overlap_frac=OVERLAP,
                            t0_frame=t0f, t1_frame=t1f)
    print(f"[{time.strftime('%T')}] KS4 epoch {EPOCH_S} -> {len(chunks)} chunks "
          f"({[ (int(c.t_start_s), int(c.t_end_s)) for c in chunks ]})", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    amps, edges = {}, []
    prev = None  # (chunk, sorting, analyzer)
    for ch in chunks:
        t0 = time.perf_counter()
        ckdir = WORK / f"chunk_{ch.index:03d}"
        bin_dir = WORK / f"chunk{ch.index:03d}_bin"
        cb = tk.materialize_chunk(BLOSC, ch, bin_dir, cmr="global", n_jobs=96)
        srt = tk.sort_chunk_ks4(cb, WORK / f"chunk{ch.index:03d}_sort", together=True)
        srt = tk.to_int_numpy_sorting(srt)
        analyzer = tk.build_chunk_analyzer(srt, cb, n_jobs=8)
        templates = tk.extract_group_templates(analyzer, cb, groups=groups)
        amps[ch.index] = {u: template_pp(t) for u, t in templates.items()}

        # checkpoint (resumable / inspectable), then drop the heavy binary
        ckdir.mkdir(parents=True, exist_ok=True)
        np.savez(ckdir / "sorting.npz", unit_ids=np.asarray(srt.unit_ids),
                 group=np.asarray(srt.get_property("group")),
                 **{f"st_{u}": srt.get_unit_spike_train(u).astype(np.int64) for u in srt.unit_ids})
        np.savez(ckdir / "templates.npz", **{f"t_{u}": templates[u] for u in srt.unit_ids})

        if prev is not None:
            pch, psrt, pan = prev
            for g in groups:
                es = tk.match_overlap(psrt, srt, pch, ch, g, match_score=0.5)
                if not es:
                    continue
                pairs = [(e["unit_a"], e["unit_b"]) for e in es]
                cos = tk.template_cosines(pan, analyzer, pairs, gci[g])
                for e in es:
                    edges.append({"group": g, "chunk_a": pch.index, "unit_a": e["unit_a"],
                                  "chunk_b": ch.index, "unit_b": e["unit_b"],
                                  "jaccard": e["jaccard"], "cosine": cos[(e["unit_a"], e["unit_b"])],
                                  "reciprocal": e["reciprocal"]})
        prev = (ch, srt, analyzer)
        shutil.rmtree(bin_dir, ignore_errors=True)
        shutil.rmtree(WORK / f"chunk{ch.index:03d}_sort", ignore_errors=True)
        n_clean = sum(1 for a in amps[ch.index].values() if a >= AMP_THR)
        print(f"[{time.strftime('%T')}] chunk {ch.index} [{int(ch.t_start_s)}-{int(ch.t_end_s)}s] "
              f"{srt.get_num_units()} units ({n_clean} clean) in {(time.perf_counter()-t0)/60:.1f} min",
              flush=True)

    (WORK / "edges.json").write_text(json.dumps(edges))
    return amps, edges, len(chunks)


def ms5_reference(n_chunks):
    """Same diagnostic for MS5 from existing checkpoints, on the aligned chunk grid."""
    global_idx = list(range(MS5_CHUNK0, MS5_CHUNK0 + n_chunks))
    amps = {}
    for k in global_idx:
        tpl_path = MS5_CHUNKS / f"chunk_{k:03d}" / "templates.npz"
        if not tpl_path.exists():
            print(f"  [warn] missing MS5 checkpoint {tpl_path}", flush=True)
            continue
        d = np.load(tpl_path)
        amps[k] = {int(key[2:]): template_pp(d[key]) for key in d.files}
    all_edges = json.loads(MS5_EDGES.read_text())
    edges = [e for e in all_edges if MS5_CHUNK0 <= e["chunk_a"] < MS5_CHUNK0 + n_chunks - 1]
    return bridge_rate(amps, edges, global_idx)


def main():
    busy = gpu_used_mb()
    print(f"[{time.strftime('%T')}] GPU memory.used = {busy:.0f} MB (near-OOM floor {GPU_FULL_MB}; "
          f"co-running shares fine)", flush=True)
    if busy > GPU_FULL_MB:
        raise SystemExit(f"ABORT: GPU nearly full ({busy:.0f} MB / 32 GB) -- not enough room "
                         f"to fit this run. Wait for the other run to finish, then re-launch.")

    rec = si.read_zarr(str(BLOSC))
    groups = sorted({int(g) for g in np.asarray(rec.get_property("group"))})
    gci = {g: tk.group_channel_indices(rec, g) for g in groups}

    t0 = time.perf_counter()
    ks4_amps, ks4_edges, n_chunks = run_ks4_chunks(rec, groups, gci)
    ks4_min = (time.perf_counter() - t0) / 60

    ks4_local = list(range(n_chunks))
    ks4_diag = bridge_rate(ks4_amps, ks4_edges, ks4_local)
    ms5_diag = ms5_reference(n_chunks)

    summary = {
        "epoch_s": EPOCH_S, "chunk_s": CHUNK_S, "overlap": OVERLAP,
        "n_chunks": n_chunks, "amp_thr_uv": AMP_THR, "jaccard_min": JACCARD_MIN,
        "ks4_minutes": round(ks4_min, 1),
        "ks4": ks4_diag, "ms5": ms5_diag,
    }
    (WORK / "bridge_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "track_ks4_chunks_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== CLEAN-UNIT PER-BOUNDARY BRIDGE RATE (epoch 36000-57600 s, 10 boundaries) ===", flush=True)
    print(f"{'sorter':<8}{'n_clean':>9}{'n_bridged':>11}{'bridge_rate':>13}{'^10':>9}{'^95':>11}", flush=True)
    for name, d in (("KS4", ks4_diag), ("MS5", ms5_diag)):
        print(f"{name:<8}{d['n_clean']:>9}{d['n_bridged']:>11}{str(d['bridge_rate']):>13}"
              f"{str(d['implied_survival_10']):>9}{str(d['implied_survival_95']):>11}", flush=True)
    print("RESULT " + json.dumps({"ks4_bridge_rate": ks4_diag["bridge_rate"],
                                  "ms5_bridge_rate": ms5_diag["bridge_rate"],
                                  "ks4_minutes": round(ks4_min, 1)}), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

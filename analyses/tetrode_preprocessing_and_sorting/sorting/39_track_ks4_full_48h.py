"""Full 48 h KS4-per-chunk tracked sort (the deconvolution counterpart to 34_track_full_48h.py).

Identical chunk+track pipeline and operating point (chunk_s=3600, overlap=0.5,
jaccard_min=0.3, cosine recorded not gated) as the MS5 production run, but each chunk is
sorted with Kilosort4 instead of MountainSort5 scheme 2. The hypothesis (gated on the 6 h
bridge test, 38_track_ks4_chunks.py): KS4's deconvolution detects every template-matching
spike, so it should not drop a clean unit between adjacent chunks the way MS5's per-chunk
clustering does (MS5 ceiling: ~0.87 clean per-boundary bridge -> long tracks fragment).
ONLY launch this if KS4 clears that ceiling by a real margin.

KS4 settings: 64-ch-together per chunk (assign_tetrode_groups recovers each unit's
tetrode by peak channel), no drift correction (nblocks=0; geometry is fictional),
templates_from_data=False (prefab universal seed templates -- avoids the no-cap KMeans
hang on dense data AND is identical across chunks, so it adds no cross-chunk
inconsistency). See tracking.sort_chunk_ks4.

Resumable, 3 phases (same as 34): A sort+checkpoint each chunk (skip any with DONE; delete
the chunk binary; torch cache cleared between chunks to keep GPU memory flat across 96
in-process KS4 runs), B load+match consecutive chunks, C chain+assemble. A crash loses at
most the in-progress chunk. Shares the host V100 with any co-running KS4 (near-OOM guard
only); under sharing the wall time is contended and not a benchmark -- the SORT is the
artifact.
"""
import gc
import json
import pathlib
import shutil
import subprocess
import time

import numpy as np
import spikeinterface as si
import torch

from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
OUT = SR / "tracked_ks4_48h"
CHUNKS_DIR = OUT / "chunks"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0

CHUNK_S = 3600.0
OVERLAP = 0.5
JACCARD_MIN = 0.3
COSINE_MIN = -1.0  # recorded, not gated
DISK_FLOOR_GB = 600
GPU_FULL_MB = 28000  # near-OOM guard only; co-running KS4 shares the 32 GB V100 fine


def gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0


def _save_chunk(sdir, srt, templates):
    sdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sdir / "sorting.npz",
        unit_ids=np.asarray(srt.unit_ids),
        group=np.asarray(srt.get_property("group")),
        **{f"st_{u}": srt.get_unit_spike_train(u).astype(np.int64) for u in srt.unit_ids},
    )
    np.savez(sdir / "templates.npz", **{f"t_{u}": templates[u].astype(np.float32) for u in templates})
    (sdir / "DONE").write_text("ok")


def _load_chunk(sdir):
    d = np.load(sdir / "sorting.npz")
    uids = d["unit_ids"]
    ud = {int(u): d[f"st_{u}"] for u in uids}
    srt = si.NumpySorting.from_unit_dict([ud], sampling_frequency=FS)
    srt.set_property("group", d["group"])
    t = np.load(sdir / "templates.npz")
    templates = {int(u): t[f"t_{u}"] for u in uids}
    return srt, templates


def main():
    si.set_global_job_kwargs(n_jobs=5, progress_bar=False, chunk_duration="1s")
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    busy = gpu_used_mb()
    print(f"[{time.strftime('%T')}] GPU memory.used = {busy:.0f} MB (near-OOM floor {GPU_FULL_MB})", flush=True)
    if busy > GPU_FULL_MB:
        raise SystemExit(f"ABORT: GPU nearly full ({busy:.0f} MB / 32 GB)")

    rec = si.read_zarr(str(BLOSC))
    n_frames = rec.get_num_frames()
    groups = sorted({int(g) for g in np.asarray(rec.get_property("group"))})
    chunks = tk.plan_chunks(n_frames, FS, chunk_s=CHUNK_S, overlap_frac=OVERLAP)
    print(f"[{time.strftime('%T')}] full KS4 sort: {len(chunks)} chunks x {len(groups)} groups "
          f"| {n_frames/FS/3600:.1f} h | /nvme free {shutil.disk_usage('/nvme').free/1e9:.0f} GB", flush=True)

    # ---- Phase A: KS4-sort + checkpoint each chunk ----
    t_a = time.perf_counter()
    for ci, ch in enumerate(chunks):
        sdir = CHUNKS_DIR / f"chunk_{ch.index:03d}"
        if (sdir / "DONE").exists():
            continue
        if shutil.disk_usage("/nvme").free / 1e9 < DISK_FLOOR_GB:
            raise SystemExit(f"ABORT: /nvme below {DISK_FLOOR_GB} GB")
        t0 = time.perf_counter()
        bin_dir = OUT / "_cur_bin"
        shutil.rmtree(bin_dir, ignore_errors=True)
        cb = tk.materialize_chunk(BLOSC, ch, bin_dir, cmr="global", n_jobs=96)
        srt = tk.to_int_numpy_sorting(tk.sort_chunk_ks4(cb, OUT / "_cur_sort", together=True))
        analyzer = tk.build_chunk_analyzer(srt, cb, n_jobs=5)
        templates = tk.extract_group_templates(analyzer, cb, groups)
        srt_n = srt.get_num_units()
        _save_chunk(sdir, srt, templates)
        shutil.rmtree(bin_dir, ignore_errors=True)
        shutil.rmtree(OUT / "_cur_sort", ignore_errors=True)
        del analyzer, srt, templates, cb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        done = ci + 1
        print(f"[{time.strftime('%T')}] chunk {ch.index} [{ch.t_start_s/3600:.1f}h] "
              f"{srt_n} units, {done}/{len(chunks)} done in {(time.perf_counter()-t0)/60:.1f} min "
              f"| GPU {gpu_used_mb():.0f} MB", flush=True)
    print(f"[{time.strftime('%T')}] Phase A done in {(time.perf_counter()-t_a)/60:.0f} min", flush=True)

    # ---- Phase B: load checkpoints; match consecutive chunks ----
    print(f"[{time.strftime('%T')}] Phase B: loading checkpoints + matching ...", flush=True)
    chunk_sortings, chunk_templates = {}, {}
    for ch in chunks:
        srt, templates = _load_chunk(CHUNKS_DIR / f"chunk_{ch.index:03d}")
        chunk_sortings[ch.index] = srt
        chunk_templates[ch.index] = templates

    edges = []
    for i in range(len(chunks) - 1):
        ca, cb_ = chunks[i], chunks[i + 1]
        for g in groups:
            spike_edges = tk.match_overlap(chunk_sortings[ca.index], chunk_sortings[cb_.index], ca, cb_, g)
            for e in spike_edges:
                ta = chunk_templates[ca.index][e["unit_a"]]
                tb = chunk_templates[cb_.index][e["unit_b"]]
                edges.append({
                    "group": g, "chunk_a": ca.index, "unit_a": int(e["unit_a"]),
                    "chunk_b": cb_.index, "unit_b": int(e["unit_b"]),
                    "jaccard": e["jaccard"], "cosine": tk.cosine_from_templates(ta, tb),
                    "reciprocal": e["reciprocal"],
                })
    print(f"[{time.strftime('%T')}] {len(edges)} candidate edges", flush=True)

    # ---- Phase C: chain + assemble ----
    node_to_global, provenance = tk.chain_matches(edges, jaccard_min=JACCARD_MIN, cosine_min=COSINE_MIN)
    global_sorting, unit_groups = tk.assemble_global_sorting(chunk_sortings, chunks, node_to_global, fs=FS)

    gdir = OUT / "global_sorting"
    np.savez(
        gdir.with_suffix(".npz"),
        unit_ids=np.asarray(global_sorting.unit_ids),
        group=np.asarray(global_sorting.get_property("group")),
        **{f"st_{u}": global_sorting.get_unit_spike_train(u).astype(np.int64) for u in global_sorting.unit_ids},
    )
    (OUT / "edges.json").write_text(json.dumps(edges, default=float))
    prov_serializable = {str(gid): [[int(m[0]), int(m[1]), int(m[2])] for m in members]
                         for gid, members in provenance.items()}
    (OUT / "provenance.json").write_text(json.dumps(prov_serializable))

    n_members = np.array([len(provenance[g]) for g in provenance])
    n_accepted = sum(1 for e in edges if e["reciprocal"] and e["jaccard"] >= JACCARD_MIN)
    summary = {
        "sorter": "kilosort4",
        "operating_point": {"chunk_s": CHUNK_S, "overlap": OVERLAP, "jaccard_min": JACCARD_MIN,
                            "cosine_min": COSINE_MIN},
        "n_chunks": len(chunks),
        "n_candidate_edges": len(edges),
        "n_accepted_edges": int(n_accepted),
        "n_global_units": int(global_sorting.get_num_units()),
        "median_chunks_per_global": float(np.median(n_members)),
        "frac_global_spanning_ge_half": round(float(np.mean(n_members >= len(chunks) / 2)), 3),
        "n_singletons": int(np.sum(n_members == 1)),
        "total_runtime_min": round((time.perf_counter() - t_a) / 60, 1),
    }
    (OUT / "tracked_ks4_48h_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "track_ks4_full_48h_summary.json").write_text(json.dumps(summary, indent=2))
    print("RESULT " + json.dumps(summary), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

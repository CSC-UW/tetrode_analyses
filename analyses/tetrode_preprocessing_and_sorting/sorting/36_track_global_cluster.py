"""Global per-tetrode clustering of unit-segments (re-runs on the checkpoints).

Consecutive chaining fragments over 48 h (even clean units bridge only ~0.91/boundary,
0.91^95->0). This swaps the matching stage for gap-tolerant global agglomeration
(tracking.heal_chains with max_gap>0): overlap-Jaccard chains are must-link anchors, and
segments separated by missed chunks are rejoined by mutual-best, unambiguous, high-cosine
template links — so a single missed boundary no longer severs a unit. Re-uses the Phase-A
checkpoints (per-chunk sorts + group templates); no re-sort.

Sweeps max_gap and validates each config: (1) do CLEAN (high-amplitude) units now span
long, and (2) FALSE-MERGE proxy = ISI refractory violations on the merged trains (a wrong
merge of two distinct units interleaves spikes -> refractory violations spike).
"""
import glob
import json
import pathlib

import numpy as np
import spikeinterface as si

from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
OUT = ROOT / "sortings_seed42_pcafix" / "tracked_48h"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0
CHUNK_S, OVERLAP, JACCARD_MIN = 3600.0, 0.5, 0.3
COSINE_MIN, MARGIN = 0.95, 0.03
MAX_GAP_GRID = [1, 2, 3]
RP_SAMPLES = int(0.0015 * FS)  # 1.5 ms refractory
CLEAN_AMP_UV = 170.0           # Q4 amplitude threshold (clean-ish units)


def _load_chunk(sdir):
    d = np.load(sdir / "sorting.npz")
    uids = d["unit_ids"]
    ud = {int(u): d[f"st_{u}"] for u in uids}
    srt = si.NumpySorting.from_unit_dict([ud], sampling_frequency=FS)
    srt.set_property("group", d["group"])
    t = np.load(sdir / "templates.npz")
    return srt, {int(u): t[f"t_{u}"] for u in uids}


def _isi_violation(train):
    if len(train) < 2:
        return 0.0
    return float(np.mean(np.diff(np.sort(train)) < RP_SAMPLES))


def _evaluate(global_sorting, provenance, chunk_templates, n_chunks):
    # amplitude per global = max member template peak-to-peak
    amp, span = {}, {}
    for gid, members in provenance.items():
        pps = [chunk_templates[c][u].max() - chunk_templates[c][u].min() for (_g, c, u) in members
               if u in chunk_templates[c]]
        amp[gid] = float(max(pps)) if pps else 0.0
        span[gid] = len({m[1] for m in members})
    gids = list(global_sorting.unit_ids)
    spans = np.array([span[g] for g in gids])
    amps = np.array([amp[g] for g in gids])
    isi = np.array([_isi_violation(global_sorting.get_unit_spike_train(g)) for g in gids])
    clean = amps >= CLEAN_AMP_UV
    long_clean = clean & (spans >= n_chunks / 2)
    return {
        "n_global": int(len(gids)),
        "median_span_all": float(np.median(spans)),
        "frac_span_ge_half": round(float(np.mean(spans >= n_chunks / 2)), 3),
        "n_clean": int(clean.sum()),
        "clean_median_span": float(np.median(spans[clean])) if clean.any() else 0.0,
        "clean_frac_span_ge_half": round(float(np.mean(spans[clean] >= n_chunks / 2)), 3) if clean.any() else 0.0,
        "clean_isi_viol_median": round(float(np.median(isi[clean])), 4) if clean.any() else 0.0,
        "longclean_isi_viol_median": round(float(np.median(isi[long_clean])), 4) if long_clean.any() else 0.0,
        "longclean_isi_viol_p90": round(float(np.percentile(isi[long_clean], 90)), 4) if long_clean.any() else 0.0,
        "n_long_clean": int(long_clean.sum()),
    }


def main():
    n_frames = si.read_zarr(str(BLOSC)).get_num_frames()
    chunks = tk.plan_chunks(n_frames, FS, chunk_s=CHUNK_S, overlap_frac=OVERLAP)
    n_chunks = len(chunks)
    chunk_sortings, chunk_templates, all_nodes = {}, {}, []
    for cd in sorted(glob.glob(str(OUT / "chunks/chunk_*"))):
        cidx = int(cd.split("_")[-1])
        srt, templates = _load_chunk(pathlib.Path(cd))
        chunk_sortings[cidx] = srt
        chunk_templates[cidx] = templates
        for u, g in zip(srt.unit_ids, np.asarray(srt.get_property("group"))):
            all_nodes.append((int(g), cidx, int(u)))
    edges = json.loads((OUT / "edges.json").read_text())
    node_to_global, provenance = tk.chain_matches(edges, jaccard_min=JACCARD_MIN, cosine_min=-1.0, nodes=all_nodes)

    base_gs, _ = tk.assemble_global_sorting(chunk_sortings, chunks, node_to_global, fs=FS)
    rows = {"consecutive_only(max_gap=none)": _evaluate(base_gs, provenance, chunk_templates, n_chunks)}

    best = None
    for mg in MAX_GAP_GRID:
        h_n2g, h_prov, h_edges = tk.heal_chains(
            node_to_global, provenance, chunk_templates, cosine_min=COSINE_MIN, margin=MARGIN, max_gap=mg)
        gs, _ = tk.assemble_global_sorting(chunk_sortings, chunks, h_n2g, fs=FS)
        ev = _evaluate(gs, h_prov, chunk_templates, n_chunks)
        ev["n_merges"] = len(h_edges)
        rows[f"global_max_gap={mg}"] = ev
        if best is None or ev["clean_frac_span_ge_half"] > best[1]["clean_frac_span_ge_half"]:
            best = (mg, ev, gs, h_prov, h_edges)

    # save the best config's global sorting
    mg, ev, gs, h_prov, h_edges = best
    np.savez(
        OUT / "global_sorting_clustered.npz",
        unit_ids=np.asarray(gs.unit_ids), group=np.asarray(gs.get_property("group")),
        **{f"st_{u}": gs.get_unit_spike_train(u).astype(np.int64) for u in gs.unit_ids},
    )
    (OUT / "provenance_clustered.json").write_text(json.dumps(
        {str(g): [[int(m[0]), int(m[1]), int(m[2])] for m in mem] for g, mem in h_prov.items()}))
    summary = {"n_chunks": n_chunks, "best_max_gap": mg, "cosine_min": COSINE_MIN,
               "clean_amp_uv": CLEAN_AMP_UV, "configs": rows}
    (OUT / "global_cluster_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "track_global_cluster_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== GLOBAL CLUSTERING SWEEP (clean = template pp >= 170 uV) ===", flush=True)
    print(f"{'config':<30}{'#glob':>7}{'med_span':>9}{'clean_med':>10}{'clean>=half':>11}{'lc_isi_med':>11}{'lc_isi_p90':>11}{'#merge':>8}", flush=True)
    for name, r in rows.items():
        print(f"{name:<30}{r['n_global']:>7}{r['median_span_all']:>9.0f}{r['clean_median_span']:>10.0f}"
              f"{r['clean_frac_span_ge_half']:>11.3f}{r['longclean_isi_viol_median']:>11.4f}"
              f"{r['longclean_isi_viol_p90']:>11.4f}{r.get('n_merges',0):>8}", flush=True)
    print("RESULT " + json.dumps({"best_max_gap": mg, **ev}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

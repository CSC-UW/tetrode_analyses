"""Heal the full-48 h tracked sort by re-linking broken chains via template cosine.

Phase A of 34_track_full_48h.py (the 4.7 h sort) is checkpointed, so this re-runs only
matching + chaining + healing (~minutes). Consecutive overlap-Jaccard chaining fragments
over 95 boundaries (per-boundary bridge ~0.82 -> ~3 h tracks); heal_chains re-links chain
ends to the next chunk's chain starts where the geometry-free 4-channel template cosine is
mutual-best, >=0.95, and an unambiguous winner. Unmatched units are retained as singletons
(so dropped continuations can be picked up). Conservative (unambiguous-only) per the
false-merge risk of template-only links on 4-channel tetrodes.
"""
import glob
import json
import pathlib
import shutil

import numpy as np
import spikeinterface as si

from tetrode_analyses import tracking as tk

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
OUT = ROOT / "sortings_seed42_pcafix" / "tracked_48h"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0
CHUNK_S, OVERLAP, JACCARD_MIN = 3600.0, 0.5, 0.3
HEAL_COSINE_MIN, HEAL_MARGIN = 0.95, 0.03


def _load_chunk(sdir):
    d = np.load(sdir / "sorting.npz")
    uids = d["unit_ids"]
    ud = {int(u): d[f"st_{u}"] for u in uids}
    srt = si.NumpySorting.from_unit_dict([ud], sampling_frequency=FS)
    srt.set_property("group", d["group"])
    t = np.load(sdir / "templates.npz")
    templates = {int(u): t[f"t_{u}"] for u in uids}
    return srt, templates


def _span_stats(provenance, n_chunks):
    chain_chunks = np.array([len({m[1] for m in members}) for members in provenance.values()])
    return {
        "n_global_units": int(len(chain_chunks)),
        "median_chunks_per_global": float(np.median(chain_chunks)),
        "frac_span_ge_half": round(float(np.mean(chain_chunks >= n_chunks / 2)), 3),
        "frac_span_all": round(float(np.mean(chain_chunks >= n_chunks)), 3),
        "max_chunks": int(chain_chunks.max()),
    }


def main():
    n_frames = si.read_zarr(str(BLOSC)).get_num_frames()
    chunks = tk.plan_chunks(n_frames, FS, chunk_s=CHUNK_S, overlap_frac=OVERLAP)
    n_chunks = len(chunks)

    # load checkpoints
    chunk_sortings, chunk_templates, all_nodes = {}, {}, []
    for cd in sorted(glob.glob(str(OUT / "chunks/chunk_*"))):
        cidx = int(cd.split("_")[-1])
        srt, templates = _load_chunk(pathlib.Path(cd))
        chunk_sortings[cidx] = srt
        chunk_templates[cidx] = templates
        groups = np.asarray(srt.get_property("group"))
        for u, g in zip(srt.unit_ids, groups):
            all_nodes.append((int(g), cidx, int(u)))
    edges = json.loads((OUT / "edges.json").read_text())
    print(f"loaded {n_chunks} chunks, {len(all_nodes)} unit-nodes, {len(edges)} edges", flush=True)

    # consecutive chaining WITH singletons retained
    node_to_global, provenance = tk.chain_matches(
        edges, jaccard_min=JACCARD_MIN, cosine_min=-1.0, nodes=all_nodes
    )
    before = _span_stats(provenance, n_chunks)

    # heal
    healed_n2g, healed_prov, heal_edges = tk.heal_chains(
        node_to_global, provenance, chunk_templates, cosine_min=HEAL_COSINE_MIN, margin=HEAL_MARGIN
    )
    after = _span_stats(healed_prov, n_chunks)

    global_sorting, unit_groups = tk.assemble_global_sorting(chunk_sortings, chunks, healed_n2g, fs=FS)

    # save healed sorting + provenance + report
    np.savez(
        OUT / "global_sorting_healed.npz",
        unit_ids=np.asarray(global_sorting.unit_ids),
        group=np.asarray(global_sorting.get_property("group")),
        **{f"st_{u}": global_sorting.get_unit_spike_train(u).astype(np.int64) for u in global_sorting.unit_ids},
    )
    (OUT / "provenance_healed.json").write_text(json.dumps(
        {str(g): [[int(m[0]), int(m[1]), int(m[2])] for m in mem] for g, mem in healed_prov.items()}))
    (OUT / "heal_edges.json").write_text(json.dumps(
        [[int(a), int(b), float(c)] for a, b, c in heal_edges]))

    summary = {
        "n_chunks": n_chunks,
        "heal_cosine_min": HEAL_COSINE_MIN, "heal_margin": HEAL_MARGIN,
        "n_heals_applied": len(heal_edges),
        "before_heal": before,
        "after_heal": after,
        "heal_cosine_median": round(float(np.median([c for _, _, c in heal_edges])), 4) if heal_edges else None,
    }
    (OUT / "heal_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "track_heal_summary.json").write_text(json.dumps(summary, indent=2))
    print("RESULT " + json.dumps(summary), flush=True)
    shutil.rmtree(OUT / "_cur_bin", ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

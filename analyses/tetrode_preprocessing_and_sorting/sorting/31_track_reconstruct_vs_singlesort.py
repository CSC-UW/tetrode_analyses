"""Validation: does chunk+track reconstruct a single drift-free sort?

On a drift-STABLE epoch (pre-discontinuity), a single MS5 scheme-2 sort is the
ground truth: with no drift, one classifier captures every well-isolated unit.
We then sort the SAME epoch in overlapping chunks and track units across chunks,
and compare the reconstructed global sorting against the single sort.

PASS GATE: among well-isolated reference units, the reconstruction's match
fraction and mean agreement are both >= 0.9. Failure here (where there is no drift
to confound) means the overlap-agreement matching signal itself is broken.

Run first at one config; 32_track_param_sweep.py then optimizes (chunk size /
overlap / thresholds).
"""
import json
import pathlib
import shutil
import time

import matplotlib
import numpy as np
import spikeinterface as si

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _track_eval as ev  # noqa: E402
from tetrode_analyses import tracking as tk  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
WORK = SR / "track_eval"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0

# Drift-stable epoch (pre-discontinuity at ~100000 s). 2 h is enough to cross
# several chunk boundaries; bump to [36000, 57600) for the definitive 6 h run.
EPOCH_S = (36000.0, 43200.0)
CHUNK_S = 1800.0
OVERLAP = 0.5
JACCARD_MIN = 0.5
COSINE_MIN = 0.9


def main():
    si.set_global_job_kwargs(n_jobs=5, progress_bar=False, chunk_duration="1s")
    WORK.mkdir(parents=True, exist_ok=True)
    t0f, t1f = int(EPOCH_S[0] * FS), int(EPOCH_S[1] * FS)
    print(f"[{time.strftime('%T')}] epoch {EPOCH_S} | chunk_s={CHUNK_S} overlap={OVERLAP} "
          f"| /nvme free {shutil.disk_usage('/nvme').free/1e9:.0f} GB", flush=True)

    # ---- reference: single scheme-2 sort of the whole epoch ----
    print(f"[{time.strftime('%T')}] building single-sort reference ...", flush=True)
    tref = time.perf_counter()
    ref_sorting, ref_an, qm_df, ref_bin = ev.single_sort_reference(
        BLOSC, WORK / "ref", t0_frame=t0f, t1_frame=t1f, fs=FS
    )
    print(f"  reference: {ref_sorting.get_num_units()} units in {(time.perf_counter()-tref)/60:.1f} min", flush=True)

    # ---- reconstruction: chunk + track over the same epoch ----
    print(f"[{time.strftime('%T')}] running chunk+track reconstruction ...", flush=True)
    trec = time.perf_counter()
    res = tk.track_span(
        BLOSC, WORK / "recon_work",
        chunk_s=CHUNK_S, overlap_frac=OVERLAP, t0_frame=t0f, t1_frame=t1f,
        jaccard_min=JACCARD_MIN, cosine_min=COSINE_MIN,
    )
    recon = res["global_sorting"]
    recon_min = (time.perf_counter() - trec) / 60
    n_chunks = len(res["chunks"])
    print(f"  reconstruction: {recon.get_num_units()} global units from {n_chunks} chunks "
          f"in {recon_min:.1f} min", flush=True)

    # ---- score ----
    score = ev.score_reconstruction(ref_sorting, qm_df, recon)

    # ---- intrinsic ceiling: per-chunk sort vs reference IN that chunk's window (NO
    # tracking). Isolates MS5 chunk-vs-whole decomposition variability from any
    # tracking loss. If recon-vs-ref ~ this ceiling, the chaining is ~lossless and
    # the limit is the sorter (cf. block-size sensitivity in SORTING_COMPARISON_FINDINGS).
    print(f"[{time.strftime('%T')}] computing intrinsic chunk-vs-whole ceiling ...", flush=True)
    ceil_fracs, ceil_ags = [], []
    for ch in res["chunks"]:
        chunk_local = res["chunk_sortings"][ch.index]
        chunk_abs = tk.shift_sorting(chunk_local, ch.start_frame)
        chunk_win = chunk_abs.frame_slice(ch.start_frame, ch.end_frame)
        ref_win = ref_sorting.frame_slice(ch.start_frame, ch.end_frame)
        cs = ev.score_reconstruction(ref_win, qm_df, chunk_win)
        ceil_fracs.append(cs["moderate"]["match_frac"])
        ceil_ags.append(cs["moderate"]["mean_agreement"])
    intrinsic_ceiling = {
        "moderate_match_frac_mean": round(float(np.nanmean(ceil_fracs)), 3),
        "moderate_mean_agreement_mean": round(float(np.nanmean(ceil_ags)), 4),
        "per_chunk_match_frac": [round(x, 3) for x in ceil_fracs],
    }
    print(f"  intrinsic ceiling (moderate): frac={intrinsic_ceiling['moderate_match_frac_mean']} "
          f"agreement={intrinsic_ceiling['moderate_mean_agreement_mean']}", flush=True)

    # tracking-specific: chunks-per-global-unit (a stable unit should span ~all chunks)
    n_members = np.array([len(res["provenance"][g]) for g in res["provenance"]])
    chunk_span_stats = {
        "n_chunks": n_chunks,
        "median_members_per_global": float(np.median(n_members)),
        "frac_global_spanning_all_chunks": round(float(np.mean(n_members == n_chunks)), 3),
        "n_singletons": int(np.sum(n_members == 1)),
    }

    good = score.get("moderate", {})
    # tracking is "lossless" if recon-vs-ref agreement reaches the intrinsic ceiling
    # (within tolerance); the absolute 0.9 gate assumes the single sort is ground
    # truth, which MS5's block-size sensitivity shows it is not.
    tracking_lossless = good.get("mean_agreement", 0) >= intrinsic_ceiling["moderate_mean_agreement_mean"] - 0.05
    passed = (good.get("match_frac", 0) >= 0.9) and (good.get("mean_agreement", 0) >= 0.9)
    summary = {
        "epoch_s": EPOCH_S,
        "chunk_s": CHUNK_S, "overlap_frac": OVERLAP,
        "jaccard_min": JACCARD_MIN, "cosine_min": COSINE_MIN,
        "reconstruction_min": round(recon_min, 1),
        "n_candidate_edges": len(res["edges"]),
        "score": score,
        "chunk_span_stats": chunk_span_stats,
        "intrinsic_ceiling": intrinsic_ceiling,
        "tracking_lossless_vs_ceiling": bool(tracking_lossless),
        "PASS_moderate_tier_absolute_0.9": bool(passed),
    }
    (OUTDIR / "track_reconstruct_vs_singlesort.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- figure: agreement of well-isolated reference units vs reconstruction ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tiers = ["all_units", "permissive", "moderate", "conservative"]
    fracs = [score[t]["match_frac"] for t in tiers]
    ags = [score[t]["mean_agreement"] for t in tiers]
    x = np.arange(len(tiers))
    ax.bar(x - 0.2, fracs, 0.4, label="match fraction")
    ax.bar(x + 0.2, ags, 0.4, label="mean agreement (matched)")
    ax.axhline(0.9, color="k", ls="--", lw=0.8, label="pass gate 0.9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={score[t]['n_ref']})" for t in tiers])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Chunk+track reconstruction vs single sort\nepoch {EPOCH_S} s, chunk={CHUNK_S}s overlap={OVERLAP}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "track_reconstruct_vs_singlesort.png", dpi=150)

    print("\n--- RECONSTRUCTION vs SINGLE SORT ---", flush=True)
    for t in tiers:
        r = score[t]
        print(f"  {t:<13} n_ref={r['n_ref']:>3} match_frac={r['match_frac']:.3f} "
              f"mean_agreement={r['mean_agreement']}", flush=True)
    print(f"chunk-span: {chunk_span_stats}", flush=True)
    print(f"intrinsic ceiling (moderate): frac={intrinsic_ceiling['moderate_match_frac_mean']} "
          f"agreement={intrinsic_ceiling['moderate_mean_agreement_mean']}", flush=True)
    print(f"RESULT recon_moderate_ag={score['moderate']['mean_agreement']} "
          f"ceiling={intrinsic_ceiling['moderate_mean_agreement_mean']} "
          f"tracking_lossless={tracking_lossless} PASS_abs0.9={passed}", flush=True)
    shutil.rmtree(WORK / "ref", ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

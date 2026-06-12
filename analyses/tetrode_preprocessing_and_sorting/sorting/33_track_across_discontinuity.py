"""Does chunk+track behave honestly across the ~100000 s discontinuity?

A waveform discontinuity sits at ~100000 s (~27.7 h, sleep deprivation; probable
electrode movement). A single 48 h scheme-2 template breaks there with no clean
merge. Here we run chunk+track over a window straddling it and ask:

1. Per-boundary BRIDGE RATE (fraction of overlap-active units that get an accepted
   reciprocal match into the next chunk). Ground-truth-free. Expectation: a DIP at
   the chunk boundary straddling 100000 s if waveforms genuinely jumped -- i.e. the
   tracker SPLITS there honestly rather than forcing a bad merge.
2. Firing-rate continuity (60 s bins) of tracked global units across the boundary,
   split by whether their chain bridges it. Contrasted with the existing 48 h
   single-template sort over the same window (which shows the discontinuity).

A split at 100000 s is a CORRECT outcome if the electrode moved; this script
quantifies it rather than judging it.
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

from tetrode_analyses import tracking as tk  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
BLOSC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
SR = ROOT / "sortings_seed42_pcafix"
WORK = SR / "track_eval"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000.0

DISCONTINUITY_S = 100000.0
WINDOW_S = (90000.0, 110000.0)  # ~5.6 h straddling the discontinuity
# operating point from the sweep (32): large chunks, low jaccard_min, cosine OFF.
# Cosine is disabled deliberately: at an electrode-movement event the waveform jumps
# while spikes stay continuous, so a strict cosine gate would wrongly break a
# same-neuron bridge. Bridging on spike-agreement alone is the correct drift behavior.
CHUNK_S = 3600.0
OVERLAP = 0.5
JACCARD_MIN = 0.3
COSINE_MIN = -1.0
SINGLE_48H = SR / "blosc-scheme2-train3600s" / "aggregated"  # for contrast


def per_boundary_bridge_rate(res):
    """For each consecutive chunk boundary, fraction of chunk-A overlap units that
    got an accepted reciprocal match into chunk B. Returns list of dicts."""
    chunks = res["chunks"]
    edges = res["edges"]
    out = []
    for i in range(len(chunks) - 1):
        ca, cb = chunks[i], chunks[i + 1]
        be = [e for e in edges if e["chunk_a"] == ca.index and e["chunk_b"] == cb.index]
        accepted = [e for e in be if e["reciprocal"] and e["jaccard"] >= JACCARD_MIN and e["cosine"] >= COSINE_MIN]
        n_a_units = len({e["unit_a"] for e in be})  # chunk-A units with any candidate in overlap
        rate = len(accepted) / n_a_units if n_a_units else float("nan")
        # the boundary "straddles" the discontinuity if it falls in this chunk pair's
        # overlap window [cb.start, ca.end]
        straddles = cb.start_frame <= DISCONTINUITY_S * FS <= ca.end_frame
        out.append({
            "boundary_mid_s": round((ca.end_frame + cb.start_frame) / 2 / FS, 1),
            "n_candidate": len(be),
            "n_accepted": len(accepted),
            "n_a_units": n_a_units,
            "bridge_rate": round(rate, 3),
            "is_discontinuity": bool(straddles),
        })
    return out


def binned_fr(spike_frames, t0_s, t1_s, bin_s=60.0):
    edges = np.arange(t0_s, t1_s + bin_s, bin_s)
    t = spike_frames / FS
    counts, _ = np.histogram(t, bins=edges)
    return edges[:-1] + bin_s / 2, counts / bin_s


def main():
    si.set_global_job_kwargs(n_jobs=5, progress_bar=False, chunk_duration="1s")
    WORK.mkdir(parents=True, exist_ok=True)
    t0f, t1f = int(WINDOW_S[0] * FS), int(WINDOW_S[1] * FS)
    print(f"[{time.strftime('%T')}] discontinuity window {WINDOW_S} | disc@{DISCONTINUITY_S}s", flush=True)

    res = tk.track_span(
        BLOSC, WORK / "disc_work",
        chunk_s=CHUNK_S, overlap_frac=OVERLAP, t0_frame=t0f, t1_frame=t1f,
        jaccard_min=JACCARD_MIN, cosine_min=COSINE_MIN,
    )
    recon = res["global_sorting"]
    chunks = res["chunks"]

    # boundary that straddles the discontinuity
    boundaries = per_boundary_bridge_rate(res)
    disc_b = [b for b in boundaries if b["is_discontinuity"]]
    other_b = [b for b in boundaries if not b["is_discontinuity"] and np.isfinite(b["bridge_rate"])]
    disc_rate = disc_b[0]["bridge_rate"] if disc_b else float("nan")
    median_other = float(np.median([b["bridge_rate"] for b in other_b])) if other_b else float("nan")

    # which global units bridge the discontinuity chunk pair (members on both sides)
    disc_chunk_idx = None
    for i in range(len(chunks) - 1):
        if chunks[i].end_frame / FS >= DISCONTINUITY_S >= chunks[i + 1].start_frame / FS:
            disc_chunk_idx = (chunks[i].index, chunks[i + 1].index)
            break
    bridged, split_like = [], []
    if disc_chunk_idx is not None:
        ia, ib = disc_chunk_idx
        for gid, members in res["provenance"].items():
            cidxs = {m[1] for m in members}
            if ia in cidxs and ib in cidxs:
                bridged.append(gid)
            elif ia in cidxs or ib in cidxs:
                split_like.append(gid)

    # ---- figure 1: bridge rate vs boundary time ----
    fig, ax = plt.subplots(figsize=(8, 4))
    bm = [b["boundary_mid_s"] for b in boundaries]
    br = [b["bridge_rate"] for b in boundaries]
    ax.plot(bm, br, "o-", color="0.4")
    for b in disc_b:
        ax.plot(b["boundary_mid_s"], b["bridge_rate"], "rs", ms=10, label="discontinuity boundary")
    ax.axvline(DISCONTINUITY_S, color="r", ls="--", lw=0.8)
    ax.set_xlabel("boundary midpoint (s)")
    ax.set_ylabel("bridge rate (accepted / overlap-active)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Cross-chunk bridge rate across discontinuity\n"
                 f"disc={disc_rate} vs median-other={median_other:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "track_discontinuity_bridge_rate.png", dpi=150)

    # ---- figure 2: FR continuity, tracked (bridged vs split) and the 48h single sort ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for gid in bridged[:25]:
        st = recon.get_unit_spike_train(gid)
        c, fr = binned_fr(st, WINDOW_S[0], WINDOW_S[1])
        axes[0].plot(c, fr, color="tab:green", alpha=0.5, lw=0.8)
    for gid in split_like[:25]:
        st = recon.get_unit_spike_train(gid)
        c, fr = binned_fr(st, WINDOW_S[0], WINDOW_S[1])
        axes[0].plot(c, fr, color="tab:red", alpha=0.4, lw=0.8)
    axes[0].axvline(DISCONTINUITY_S, color="r", ls="--", lw=0.8)
    axes[0].set_ylabel("firing rate (Hz)")
    axes[0].set_title(f"chunk+track units (green=bridged n={len(bridged)}, red=split-like n={len(split_like)})")

    contrast_note = "n/a"
    if SINGLE_48H.exists():
        s48 = si.load(str(SINGLE_48H)).frame_slice(t0f, t1f)
        for u in s48.unit_ids[:40]:
            st = s48.get_unit_spike_train(u)
            c, fr = binned_fr(st, WINDOW_S[0], WINDOW_S[1])
            axes[1].plot(c, fr, color="0.5", alpha=0.4, lw=0.8)
        axes[1].set_title(f"48 h single-template scheme-2 sort, same window (n={s48.get_num_units()} units)")
        contrast_note = f"{s48.get_num_units()} units"
    axes[1].axvline(DISCONTINUITY_S, color="r", ls="--", lw=0.8)
    axes[1].set_ylabel("firing rate (Hz)")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "track_discontinuity_fr_continuity.png", dpi=150)

    summary = {
        "window_s": WINDOW_S, "discontinuity_s": DISCONTINUITY_S,
        "chunk_s": CHUNK_S, "overlap": OVERLAP,
        "n_global_units": recon.get_num_units(),
        "discontinuity_boundary_chunks": disc_chunk_idx,
        "bridge_rate_at_discontinuity": disc_rate,
        "median_bridge_rate_other_boundaries": round(median_other, 3),
        "n_bridged_across_discontinuity": len(bridged),
        "n_split_like_at_discontinuity": len(split_like),
        "per_boundary": boundaries,
        "single_48h_contrast": contrast_note,
    }
    (OUTDIR / "track_across_discontinuity.json").write_text(json.dumps(summary, indent=2, default=float))
    print("RESULT " + json.dumps({k: summary[k] for k in (
        "bridge_rate_at_discontinuity", "median_bridge_rate_other_boundaries",
        "n_bridged_across_discontinuity", "n_split_like_at_discontinuity")}), flush=True)
    shutil.rmtree(WORK / "disc_work", ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

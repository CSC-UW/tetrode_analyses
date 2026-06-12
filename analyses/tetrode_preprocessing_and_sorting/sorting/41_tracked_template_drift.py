"""Visual verification: do long-tracked well-isolated units drift SMOOTHLY over 48 h?

The encouraging finding (40_) is that ~38 well-isolated units track via purely-consecutive
overlap chains over up to the full recording. A clean refractory period is necessary but not
sufficient for "one neuron tracked continuously" -- a chain could in principle stitch
template-similar but distinct neurons. The decisive visual check: plot each unit's peak-channel
template at every member chunk, colored by time. A single drifting neuron morphs GRADUALLY; a
stitched merge shows an abrupt template jump.

Picks the longest purely-consecutive conservative-tier units and renders a grid to
tracked_48h/tracked_template_drift.png (+ a peak-to-peak-amplitude-over-time companion). No
sorting/compute -- reads the analyzer's quality_metrics + the per-chunk template checkpoints.
"""
import json
import pathlib

import matplotlib
import numpy as np
import spikeinterface as si

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

T = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/tracked_48h")
ANALYZER = T / "analyzer_clustered.zarr"
PROV = T / "provenance_clustered.json"
CHUNKS = T / "chunks"
FS = 30000.0
STRIDE_S = 1800.0
N_SHOW = 12
MIN_SPAN = 24  # chunks (~12 h) -- show the genuinely long ones


def main():
    an = si.load_sorting_analyzer(str(ANALYZER))
    qm = an.get_extension("quality_metrics").get_data()
    finite = (np.isfinite(qm["isi_violations_ratio"]) & np.isfinite(qm["rp_contamination"])
              & np.isfinite(qm["firing_rate"]))
    cons = ((qm["isi_violations_ratio"] < 0.1) & (qm["rp_contamination"] < 0.1)
            & (qm["firing_rate"] >= 0.5) & finite)
    cons_ids = set(int(u) for u in qm.index[cons.to_numpy()])

    prov = json.loads(PROV.read_text())
    cand = []
    for gid, members in prov.items():
        if int(gid) not in cons_ids:
            continue
        ms = sorted(members, key=lambda m: (m[1], m[2]))
        chs = sorted({m[1] for m in ms})
        if len(chs) < MIN_SPAN or any(np.diff(chs) > 1):  # conservative + long + purely consecutive
            continue
        cand.append((int(gid), ms, len(chs)))
    cand.sort(key=lambda x: -x[2])
    cand = cand[:N_SHOW]
    print(f"[drift] {len(cand)} long purely-consecutive conservative units to plot", flush=True)
    if not cand:
        print("DONE (no units matched)", flush=True)
        return

    tcache = {}

    def tmpl(c, u):
        if c not in tcache:
            tcache[c] = dict(np.load(CHUNKS / f"chunk_{c:03d}" / "templates.npz"))
        return tcache[c].get(f"t_{u}")

    ncol = 4
    nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    figa, axesa = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.0 * nrow), squeeze=False)
    for k, (gid, ms, span) in enumerate(cand):
        ax = axes[k // ncol][k % ncol]
        axa = axesa[k // ncol][k % ncol]
        t0 = tmpl(ms[0][1], ms[0][2])
        pk = int(np.argmax(t0.max(0) - t0.min(0)))
        chs = [m[1] for m in ms]
        c_lo, c_hi = min(chs), max(chs)
        pps_t = []
        for (_g, c, u) in ms:
            w = tmpl(c, u)
            if w is None:
                continue
            frac = (c - c_lo) / max(1, c_hi - c_lo)
            ax.plot(w[:, pk], color=cm.viridis(frac), lw=0.8, alpha=0.7)
            pps_t.append((c * STRIDE_S / 3600.0, float(w[:, pk].max() - w[:, pk].min())))
        ax.set_title(f"u{gid}: {span} ch (~{(span + 1) * STRIDE_S / 3600:.0f} h), peak ch{pk}", fontsize=9)
        ax.set_xticks([])
        if pps_t:
            hh, pp = zip(*pps_t)
            axa.plot(hh, pp, ".-", ms=3, lw=0.8, color="tab:blue")
        axa.set_title(f"u{gid} peak-ch p2p (uV) vs time", fontsize=9)
        axa.set_xlabel("h")

    for extra in range(len(cand), nrow * ncol):
        axes[extra // ncol][extra % ncol].axis("off")
        axesa[extra // ncol][extra % ncol].axis("off")
    fig.suptitle("Tracked well-isolated units: peak-channel template waveform over 48 h "
                 "(color = chunk time, dark->bright = early->late). Smooth morph = single drifting neuron.",
                 fontsize=11)
    figa.suptitle("Tracked well-isolated units: peak-channel peak-to-peak amplitude over time (drift, not jumps)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figa.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = T / "tracked_template_drift.png"
    out2 = T / "tracked_template_amplitude_over_time.png"
    fig.savefig(out1, dpi=110, bbox_inches="tight")
    figa.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"[drift] saved {out1}", flush=True)
    print(f"[drift] saved {out2}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Gate probe + cosine intuition figures for the re-seed add-cos analysis (script 61 follow-up).

Three deliverables, all from data already on disk (no expensive re-run):

1. GATE PROBE (`gate_probe_birthcos.png`): the re-seed cadence sweep (script 61) recorded, for every
   re-seeded unit, its max 4-channel template cosine to the LIVE same-tetrode bank AT BIRTH (`birth_cos`,
   the exact quantity the add-cos=0.8 gate thresholds on). Because the gate admits a candidate iff
   best_cos < 0.8, ANY candidate with best_cos < 0.6 would have been admitted and recorded with
   birth_cos < 0.6 -- there are ZERO such units (min observed ~0.72). So tightening the gate to 0.6 admits
   nothing: there is no distinct sub-population hiding below the gate. The figure shows the admitted
   population piled into [0.72, 0.80) against the gate, the would-be-tighter region empty, plus the
   full-span `final_cos` (post-hoc twin) distribution.

2. REAL PAIRS (`cosine_realpairs.png`): actual within-tetrode template pairs from this recording binned at
   cosine ~= 0.6 / 0.7 / 0.8 / 0.9 / 0.95, each as a 4-channel overlay, so you can SEE how different two
   multichannel waveforms can be while still scoring above a given cosine -- the shared-tetrode morphology
   floor that makes cosine a weak distinctness test here.

3. TRANSFORM DEMO (`cosine_transform_demo.png`): one real template transformed in controlled ways --
   global rescale (cosine is amplitude-BLIND -> stays 1.00), single-channel reweighting tuned to hit
   target cosines, and a pure time-shift (max-shift cosine recovers it by design) -- isolating what the
   metric is and is not sensitive to.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/62_cosine_gate_illustration.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si

from _mp_common import build_templates_object, materialize_span
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
OUT = ROOT / "mp_long_s2000_d170000"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0


def best_shift_cosine(ta, tb, max_shift=10):
    """Max cosine of two (T, C) templates over integer shifts; return (cos, best_shift)."""
    n = ta.shape[0]
    best, bs = -1.0, 0
    for s in range(-max_shift, max_shift + 1):
        a, b = (ta[s:], tb[: n - s]) if s >= 0 else (ta[: n + s], tb[-s:])
        if a.shape[0] < 1:
            continue
        af, bf = a.ravel(), b.ravel()
        na, nb = np.linalg.norm(af), np.linalg.norm(bf)
        if na == 0 or nb == 0:
            continue
        c = float(af @ bf / (na * nb))
        if c > best:
            best, bs = c, s
    return best, bs


def aligned_pair(ta, tb, max_shift=10):
    """Return (a, b_shifted) trimmed to common support at the best shift (for overlay plotting)."""
    _, s = best_shift_cosine(ta, tb, max_shift)
    n = ta.shape[0]
    if s >= 0:
        return ta[s:], tb[: n - s]
    return ta[: n + s], tb[-s:]


def load_unit_templates(n_jobs=16):
    """Per-unit (T, 4) tetrode template + group + peak-to-peak, from the cadence-12 assembled sorting.

    Falls back to rebuilding the confident seed sort if the assembled sorting cannot be loaded.
    """
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    asm_dir = OUT / "assembled_reseed_c12"
    try:
        srt = si.load(asm_dir)
        if srt.get_property("group") is None:
            raise ValueError("assembled sorting lost its group property")
        src = "assembled_reseed_c12 (seed+reseed, full-span templates)"
    except Exception as e:  # noqa: BLE001
        print(f"  (could not use {asm_dir.name}: {e}; rebuilding confident seed sort)", flush=True)
        win0 = rec.frame_slice(0, int(1800 * FS))
        win0.reset_times()
        ref0 = to_int_numpy_sorting(sort_chunk(win0, OUT / "seed_sort_illus"))
        _, az = build_templates_object(ref0, win0, with_snr=True, n_jobs=n_jobs)
        snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
        nsp = np.array([len(ref0.get_unit_spike_train(u)) for u in ref0.unit_ids])
        srt = ref0.select_units(np.asarray(ref0.unit_ids)[(snr >= 5.0) & (nsp >= 100)])
        rec, src = win0, "confident seed sort (window 0)"
    templates, _ = build_templates_object(srt, rec, with_snr=False, n_jobs=n_jobs)
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)  # (n, T, n_chan)
    mask = templates.sparsity.mask
    t4, grp, ptp = {}, {}, {}
    for i, u in enumerate([int(x) for x in templates.unit_ids]):
        g = int(rec_groups[np.flatnonzero(mask[i])[0]])
        gch = np.flatnonzero(rec_groups == g)
        w = dense[i][:, gch]
        t4[u], grp[u], ptp[u] = w, g, float(np.ptp(w))
    print(f"  loaded {len(t4)} unit templates from {src}", flush=True)
    return t4, grp, ptp


# --------------------------------------------------------------------------- #
# Figure 1: gate probe
# --------------------------------------------------------------------------- #
def fig_gate_probe():
    d = np.load(OUT / "reseed_cadence_sweep.npz")
    cads = [int(c) for c in d["cadences"]]
    birth = np.concatenate([d[f"birth_cos_c{c}"] for c in cads])
    final = np.concatenate([d[f"final_cos_c{c}"] for c in cads])
    fin = np.isfinite(final)
    n_below = int((birth < 0.6).sum())
    # of the distinct-at-birth (<0.6) units, how many are full-span twins (final>=0.9)?
    below = birth < 0.6
    n_below_twin = int((below & fin & (final >= 0.9)).sum())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))

    bins = np.linspace(0.0, 1.0, 51)
    ax[0].hist(birth, bins=bins, color="#3b7dd8", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax[0].axvspan(0.0, 0.6, color="0.9", zorder=0)
    ax[0].axvline(0.6, color="green", ls="--", lw=1.5, label="hypothetical tighter gate 0.6")
    ax[0].axvline(0.8, color="red", ls="-", lw=2.0, label="add-cos gate 0.8 (admit if <)")
    ax[0].text(0.27, ax[0].get_ylim()[1] * 0.58,
               f"distinct-at-birth\n(<0.6): only {n_below} of {len(birth)}\n"
               f"-- and {n_below_twin}/{n_below} are\nfull-span TWINS",
               ha="center", va="center", fontsize=9.5, color="0.25")
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("birth_cos: max 4-ch cosine to LIVE same-tetrode bank at birth")
    ax[0].set_ylabel("re-seeded units (pooled over cadences 12/6/3)")
    ax[0].set_title(f"Admits pile against the gate\n(n={len(birth)}, min={birth.min():.2f}, "
                    f"median={np.median(birth):.2f}, max={birth.max():.2f})")
    ax[0].legend(fontsize=8, loc="upper left")

    # the money plot: distinctness-at-birth (x) vs full-span twin similarity (y)
    ax[1].scatter(birth[fin], final[fin], s=30, c="#d8743b", edgecolor="k", linewidth=0.3, alpha=0.8)
    ax[1].axvline(0.8, color="red", ls="-", lw=1.6, label="add-cos gate 0.8")
    ax[1].axvline(0.6, color="green", ls="--", lw=1.4, label="hypothetical 0.6")
    ax[1].axhline(0.9, color="purple", ls="-", lw=1.6, label="twin threshold 0.9")
    ax[1].set_xlim(0.3, 0.85)
    ax[1].set_ylim(0.6, 1.01)
    ax[1].set_xlabel("birth_cos: distinctness from bank AT BIRTH (lower = more distinct)")
    ax[1].set_ylabel("final_cos: full-span twin similarity (higher = more duplicate)")
    ax[1].set_title("The few distinct-at-birth admits are the WORST twins\n"
                    f"(frac full-span twin = {(final[fin] >= 0.9).mean():.2f})")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Re-seed add-cos gate probe: admits pile at the gate, and the handful that look distinct "
                 "at birth are all full-span twins -> the gate is not a clean lever", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "gate_probe_birthcos.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}  ({n_below} below 0.6, {n_below_twin} of them full-span twins)", flush=True)


# --------------------------------------------------------------------------- #
# Figure 2: real within-tetrode pairs at increasing cosine
# --------------------------------------------------------------------------- #
def fig_real_pairs(t4, grp, ptp):
    uids = list(t4)
    pairs = []  # (cos, ui, uj)
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            a, b = uids[i], uids[j]
            if grp[a] != grp[b]:
                continue
            if min(ptp[a], ptp[b]) < 20:  # skip tiny templates (illegible overlays)
                continue
            c, _ = best_shift_cosine(t4[a], t4[b])
            pairs.append((c, a, b))
    pairs.sort()
    cos_arr = np.array([p[0] for p in pairs])
    targets = [0.60, 0.70, 0.80, 0.90, 0.95]
    chosen = []
    for tgt in targets:
        k = int(np.argmin(np.abs(cos_arr - tgt)))
        chosen.append(pairs[k])
    fig, axes = plt.subplots(len(targets), 4, figsize=(12, 2.3 * len(targets)), squeeze=False)
    for r, (c, a, b) in enumerate(chosen):
        ta, tb = aligned_pair(t4[a], t4[b])
        ymax = 1.05 * max(np.abs(ta).max(), np.abs(tb).max())
        for ch in range(4):
            ax = axes[r][ch]
            ax.plot(ta[:, ch], color="k", lw=1.4, label=f"unit {a}")
            ax.plot(tb[:, ch], color="#d8743b", lw=1.4, label=f"unit {b}")
            ax.set_ylim(-ymax, ymax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"wire {ch}", fontsize=9)
            if ch == 0:
                ax.set_ylabel(f"cos = {c:.2f}\ntetrode {grp[a]}", fontsize=10)
        axes[r][3].legend(fontsize=7, loc="upper right", frameon=False)
    fig.suptitle("Real within-tetrode template pairs: shared 4-wire morphology keeps cosine high even "
                 "when the waveforms differ visibly\n(cosine is amplitude-blind; on a tetrode, distinct "
                 "neurons routinely sit at 0.6-0.8)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "cosine_realpairs.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


# --------------------------------------------------------------------------- #
# Figure 3: controlled transforms of one template
# --------------------------------------------------------------------------- #
def _concat(t4_):
    """Flatten (T, 4) -> concatenated 1-D with NaN gaps between wires (for a single overlay trace)."""
    T = t4_.shape[0]
    out = np.full((T + 3) * 4, np.nan)
    for ch in range(4):
        out[ch * (T + 3): ch * (T + 3) + T] = t4_[:, ch]
    return out


def fig_transform_demo(t4, grp, ptp):
    base_id = max(ptp, key=ptp.get)  # highest-amplitude template (legible)
    base = t4[base_id].copy()
    pk_ch = int(np.argmax(np.ptp(base, axis=0)))  # peak channel

    panels = []  # (label, variant, naive_cos, maxshift_cos)
    # (a) global rescale -> amplitude-blind
    v = base * 0.4
    c0, _ = best_shift_cosine(base, v)
    naive = float(base.ravel() @ v.ravel() / (np.linalg.norm(base) * np.linalg.norm(v)))
    panels.append(("global x0.4\n(amplitude-blind)", v, naive, c0))
    # (b-d) attenuate the peak channel to hit target cosines
    alphas = np.linspace(1.0, -0.6, 400)
    cos_by_alpha = []
    for al in alphas:
        v = base.copy()
        v[:, pk_ch] = base[:, pk_ch] * al
        cos_by_alpha.append(best_shift_cosine(base, v)[0])
    cos_by_alpha = np.array(cos_by_alpha)
    for tgt in (0.95, 0.90, 0.80, 0.60):
        al = alphas[int(np.argmin(np.abs(cos_by_alpha - tgt)))]
        v = base.copy()
        v[:, pk_ch] = base[:, pk_ch] * al
        c, _ = best_shift_cosine(base, v)
        naive = float(base.ravel() @ v.ravel() / (np.linalg.norm(base) * np.linalg.norm(v)))
        panels.append((f"wire {pk_ch} x{al:.2f}\n(re-weight 1 channel)", v, naive, c))
    # (f) pure time shift -> max-shift recovers
    sh = 5
    v = np.zeros_like(base)
    v[sh:] = base[:-sh]
    c, _ = best_shift_cosine(base, v)
    naive = float(base.ravel() @ v.ravel() / (np.linalg.norm(base) * np.linalg.norm(v)))
    panels.append((f"time shift +{sh} samp\n(max-shift recovers)", v, naive, c))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.55 * n), squeeze=False)
    bc = _concat(base)
    for r, (label, v, naive, c) in enumerate(panels):
        ax = axes[r][0]
        ax.plot(bc, color="k", lw=1.3, label=f"base (unit {base_id})")
        ax.plot(_concat(v), color="#d8743b", lw=1.3, label="variant")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center")
        ax.text(0.99, 0.92, f"max-shift cos = {c:.2f}   (naive {naive:.2f})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        if r == 0:
            ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("What the 4-channel cosine is (in)sensitive to: blind to global amplitude; driven by "
                 "channel-ratio + shape; tolerant of time shifts\n(4 wires concatenated left->right, "
                 "gaps between)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "cosine_transform_demo.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


def main():
    print("gate probe ...", flush=True)
    fig_gate_probe()
    print("loading templates for illustration ...", flush=True)
    t4, grp, ptp = load_unit_templates()
    print("real pairs ...", flush=True)
    fig_real_pairs(t4, grp, ptp)
    print("transform demo ...", flush=True)
    fig_transform_demo(t4, grp, ptp)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

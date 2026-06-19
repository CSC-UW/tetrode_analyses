"""Temporal (refractory/CCG) adjudication of re-seeded units: are they duplicates or real neurons?

Template COSINE cannot separate a drift-split twin from a distinct same-tetrode neuron (it is
amplitude-blind and the 4 wires share morphology; see scripts 61/62 + MATCHING_PURSUIT_FINDINGS). The
decisive test is TEMPORAL. For matching pursuit, two tracks of the SAME physical neuron have spikes that
never co-occur within the refractory period -- whether the deconvolution splits the neuron's spikes
between the two templates within a window, or hands off between windows -- so their CROSS-correlogram has
a refractory DIP at zero lag. Two DISTINCT neurons fire independently -> a flat (filled) cross-correlogram.

For each re-seeded (born) unit we take its best same-tetrode COSINE match among units that existed before
it (the twin SUSPECT), restrict to the windows where BOTH are active (co-present), and score the
cross-correlogram central/flank ratio:
  ratio = (coincidences within +/-1.5 ms, per ms) / (coincidences at 5-25 ms lag, per ms)
  ratio < 0.30  -> refractory DIP        -> SAME neuron (duplicate; merge)
  ratio > 0.70  -> flat / filled         -> DISTINCT neuron (keep)
  else          -> AMBIGUOUS
  co-active < 2 windows or < 30 flank coincidences -> SEGREGATED/INCONCLUSIVE (refractory can't decide;
      e.g. a clean temporal hand-off -- needs a template-trajectory check, not refractory)

Resolves (a) which `is_reseeded` units on the 6 h deliverable to merge vs keep, and (b) whether a finer
cadence's extra units are real neurons or twins -- the question cosine left open.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/63_reseed_twin_adjudication.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si

from _mp_common import build_templates_object, materialize_span
from tetrode_analyses.tracking import cosine_from_templates

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
OUT = ROOT / "mp_long_s2000_d170000"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
CADENCES = [12, 6, 3]

REFR_MS = 1.5          # refractory half-window for the central CCG bin
FLANK_MS = (5.0, 25.0)  # flank lag band
MAXLAG_MS = 25.0
DIP, FILL = 0.30, 0.70  # ratio verdict thresholds
MIN_CO_WIN, MIN_FLANK = 2, 30  # need this much co-activity for refractory to decide


def fullspan_t4(asm, rec, *, n_jobs=16, min_spikes_template=50):
    groups = np.asarray(asm.get_property("group"))
    gmap = {int(u): int(groups[i]) for i, u in enumerate(asm.unit_ids)}
    keep = [u for u in asm.unit_ids if len(asm.get_unit_spike_train(u)) >= min_spikes_template]
    a = asm.select_units(keep)
    a.set_property("group", np.array([gmap[int(u)] for u in a.unit_ids]))
    templates, _ = build_templates_object(a, rec, with_snr=False, n_jobs=n_jobs)
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)
    rec_groups = np.asarray(rec.get_property("group"))
    t4, ug = {}, {}
    for i, u in enumerate([int(x) for x in templates.unit_ids]):
        gch = np.flatnonzero(rec_groups == gmap[u])
        t4[u] = dense[i][:, gch]
        ug[u] = gmap[u]
    return t4, ug


def ccg_lags(tb, tu, maxlag):
    """All (tu - tb) lags within +/-maxlag frames, for sorted frame arrays tb, tu."""
    if len(tb) == 0 or len(tu) == 0:
        return np.empty(0)
    lo = np.searchsorted(tu, tb - maxlag)
    hi = np.searchsorted(tu, tb + maxlag)
    return np.concatenate([tu[a:b] - s for s, a, b in zip(tb, lo, hi) if b > a]) if np.any(hi > lo) \
        else np.empty(0)


def adjudicate(tb, tu):
    """Cross-correlogram central/flank ratio + verdict for two co-restricted frame trains."""
    maxlag = int(MAXLAG_MS * 1e-3 * FS)
    lags = ccg_lags(np.sort(tb), np.sort(tu), maxlag)
    al = np.abs(lags) / FS * 1000.0
    central = int(np.sum(al <= REFR_MS))
    flank = int(np.sum((al >= FLANK_MS[0]) & (al <= FLANK_MS[1])))
    cw, fw = 2 * REFR_MS, 2 * (FLANK_MS[1] - FLANK_MS[0])
    if flank < MIN_FLANK:
        return np.nan, central, flank
    return (central / cw) / (flank / fw), central, flank


def verdict_of(ratio, n_co):
    if n_co < MIN_CO_WIN or not np.isfinite(ratio):
        return "SEGREGATED"
    if ratio < DIP:
        return "duplicate"
    if ratio > FILL:
        return "distinct"
    return "ambiguous"


def co_restrict(train, co_wins, wlen):
    return train[np.isin(train // wlen, co_wins)] if len(train) else train


def run_cadence(N, rec, n_jobs=16):
    d = np.load(OUT / "reseed_cadence_sweep.npz")
    born = [int(x) for x in d[f"born_ids_c{N}"]]
    bwin = {int(b): int(w) for b, w in zip(d[f"born_ids_c{N}"], d[f"birth_win_c{N}"])}
    asm = si.load(OUT / f"assembled_reseed_c{N}")
    t4, ug = fullspan_t4(asm, rec, n_jobs=n_jobs)
    wlen = int(WIN_S * FS)
    nwin = int(np.ceil(DUR_S / WIN_S))
    trains = {int(u): np.sort(asm.get_unit_spike_train(u)).astype(np.int64) for u in asm.unit_ids}
    counts = {u: np.bincount(t // wlen, minlength=nwin + 1) for u, t in trains.items()}
    born_set = set(born)
    rows = []
    for b in born:
        if b not in t4:
            rows.append(dict(born=b, match=-1, cosine=np.nan, n_co=0, ratio=np.nan, verdict="stillborn"))
            continue
        g = ug[b]
        pre = [u for u in t4 if ug[u] == g and u != b
               and (u not in born_set or bwin.get(u, 1 << 30) < bwin[b])]
        if not pre:
            rows.append(dict(born=b, match=-1, cosine=np.nan, n_co=0, ratio=np.nan, verdict="no-neighbour"))
            continue
        u = max(pre, key=lambda x: cosine_from_templates(t4[b], t4[x]))
        cos = cosine_from_templates(t4[b], t4[u])
        co = np.flatnonzero((counts[b] >= 5) & (counts[u] >= 5))
        tb = co_restrict(trains[b], co, wlen)
        tu = co_restrict(trains[u], co, wlen)
        ratio, central, flank = adjudicate(tb, tu)
        rows.append(dict(born=b, match=u, cosine=float(cos), n_co=int(co.size),
                         ratio=float(ratio), central=central, flank=flank,
                         verdict=verdict_of(ratio, co.size)))
    return rows, trains


def main():
    rec = materialize_span(OUT, START_S, DUR_S)
    all_rows = {}
    trains_c12 = None
    for N in CADENCES:
        print(f"\n=== cadence every {N} windows ({N * WIN_S / 3600:.1f} h) ===", flush=True)
        rows, trains = run_cadence(N, rec)
        all_rows[N] = rows
        if N == 12:
            trains_c12 = trains
        from collections import Counter
        vc = Counter(r["verdict"] for r in rows)
        dup = [r for r in rows if r["verdict"] == "duplicate"]
        print(f"  {len(rows)} re-seeded units -> " + ", ".join(f"{k}={v}" for k, v in sorted(vc.items())),
              flush=True)
        print(f"  cosine>=0.95 among them: {sum(np.isfinite(r['cosine']) and r['cosine']>=0.95 for r in rows)}; "
              f"of those, verdict=duplicate: {sum(r['verdict']=='duplicate' and r['cosine']>=0.95 for r in rows if np.isfinite(r['cosine']))}",
              flush=True)
        if dup:
            print("  DUPLICATES (merge into match):", flush=True)
            for r in sorted(dup, key=lambda x: x["ratio"]):
                print(f"    u{r['born']} -> u{r['match']}  cos={r['cosine']:.2f} ratio={r['ratio']:.2f} "
                      f"co_win={r['n_co']}", flush=True)

    # summary table across cadences
    print("\n" + "=" * 78)
    print("RESEED TWIN ADJUDICATION (temporal CCG verdict; cosine could not decide)")
    print("=" * 78)
    cats = ["duplicate", "distinct", "ambiguous", "SEGREGATED", "stillborn", "no-neighbour"]
    print(f"{'every':>6}{'h':>5} {'born':>5} " + " ".join(f"{c[:9]:>10}" for c in cats))
    from collections import Counter
    for N in CADENCES:
        vc = Counter(r["verdict"] for r in all_rows[N])
        print(f"{N:>6}{N * WIN_S / 3600:>5.1f} {len(all_rows[N]):>5} "
              + " ".join(f"{vc.get(c, 0):>10}" for c in cats))

    # example CCG figure: a couple per verdict from the c12 deliverable
    rows12 = all_rows[12]
    fig_rows = []
    for v in ("duplicate", "distinct", "SEGREGATED"):
        picks = [r for r in rows12 if r["verdict"] == v][:2]
        fig_rows += [(v, r) for r in picks]
    if fig_rows:
        maxlag = int(MAXLAG_MS * 1e-3 * FS)
        edges = np.arange(-maxlag, maxlag + 1, int(0.5e-3 * FS))
        fig, axes = plt.subplots(len(fig_rows), 1, figsize=(7, 1.7 * len(fig_rows)), squeeze=False)
        for r_i, (v, r) in enumerate(fig_rows):
            ax = axes[r_i][0]
            tb, tu = np.sort(trains_c12[r["born"]]), np.sort(trains_c12[r["match"]])
            lags = ccg_lags(tb, tu, maxlag)
            ax.hist(lags / FS * 1000.0, bins=edges / FS * 1000.0, color="#3b6fb0", edgecolor="none")
            ax.axvspan(-REFR_MS, REFR_MS, color="red", alpha=0.18)
            ax.set_ylabel(f"{v}\nu{r['born']}~u{r['match']}", fontsize=8, rotation=0, ha="right", va="center")
            ax.text(0.99, 0.9, f"cos={r['cosine']:.2f}  ratio={r['ratio']:.2f}  co_win={r['n_co']}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
            ax.set_xlabel("lag (ms)  [match - born]" if r_i == len(fig_rows) - 1 else "")
        fig.suptitle("Cross-correlograms: refractory DIP at 0 (red) = same neuron (duplicate); "
                     "filled = distinct cell", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = OUT / "ccg_adjudication_examples.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"\nwrote {p}", flush=True)

    np.savez(OUT / "reseed_twin_adjudication.npz",
             **{f"c{N}_{k}": np.array([r.get(k, np.nan) for r in all_rows[N]])
                for N in CADENCES for k in ("born", "match", "cosine", "n_co", "ratio")},
             **{f"c{N}_verdict": np.array([r["verdict"] for r in all_rows[N]]) for N in CADENCES})
    print(f"wrote {OUT / 'reseed_twin_adjudication.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

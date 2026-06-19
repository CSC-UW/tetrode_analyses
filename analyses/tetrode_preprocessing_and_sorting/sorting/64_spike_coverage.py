"""What fraction of detectable spikes does our sorting account for? (detection completeness)

A sorting's "false-negative" rate has no ground truth, so we use the standard proxy: run an INDEPENDENT,
sorter-agnostic reference detector (locally-exclusive negative peaks at 5.5 MAD, per tetrode) over the
whole span -> the set of spike-like EVENTS. (This is NOT literally MS5's front-end nor part of the
matching-pursuit matchers, which do no threshold detection; it is a common yardstick chosen to resemble a
generic detector.) A sorted spike "claims" an event if it lands within +/-0.5 ms of it (same tetrode). The
unclaimed fraction is the false-negative proxy. radius_um=40 groups each tetrode's 4 wires (<=20 um apart
in the fictional probegroup) into one neighborhood while never bridging tetrodes (300 um apart), so each
spike yields one event on its dominant channel.

The decisive cut is AMPLITUDE: unclaimed events at ~5.5-7 MAD are mostly multi-unit hash / noise that no
sorter should (or can) isolate -- expected, not a problem. Unclaimed events at high amplitude (e.g. >10
MAD) are large, clearly-real spikes belonging to a unit the bank is MISSING -- the striking false
negatives. We report the claimed fraction in amplitude bins and as a curve, for the re-seed deliverable
(bank = seed + reseeds) and the no-reseed deliverable (delta = coverage re-seeding actually adds).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/64_spike_coverage.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si
from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from _mp_common import materialize_span

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
TOL_MS = 0.5
DETECT_THRESH = 5.5
SORTINGS = [("reseed (seed+reseeds)", "assembled_reseed_rs"),
            ("no-reseed (seed only)", "assembled_reestimate_dedup095")]
AMP_BINS = [5.5, 7, 9, 12, 16, 24, np.inf]


def per_tetrode_sorted(sorting, rec_groups):
    """{tetrode group: sorted np.int64 spike frames pooled over that tetrode's units}."""
    grp = np.asarray(sorting.get_property("group"))
    out = {}
    for i, u in enumerate(sorting.unit_ids):
        g = int(grp[i])
        out.setdefault(g, []).append(sorting.get_unit_spike_train(u).astype(np.int64))
    return {g: np.sort(np.concatenate(v)) for g, v in out.items()}, int(sum(len(t) for t in
            (sorting.get_unit_spike_train(u) for u in sorting.unit_ids)))


def claimed_mask(peak_s, peak_g, sorted_by_tet, tol):
    """Bool per peak: is there a sorted spike within +/-tol frames on the SAME tetrode?"""
    out = np.zeros(len(peak_s), dtype=bool)
    for g in np.unique(peak_g):
        st = sorted_by_tet.get(int(g))
        sel = peak_g == g
        if st is None or st.size == 0:
            continue
        ps = peak_s[sel]
        j = np.searchsorted(st, ps)
        dprev = np.where(j > 0, ps - st[np.clip(j - 1, 0, st.size - 1)], tol + 1)
        dnext = np.where(j < st.size, st[np.clip(j, 0, st.size - 1)] - ps, tol + 1)
        out[np.flatnonzero(sel)] = np.minimum(dprev, dnext) <= tol
    return out


def main():
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    tol = int(TOL_MS * 1e-3 * FS)
    print(f"detecting threshold crossings (locally_exclusive, neg, {DETECT_THRESH} MAD) over "
          f"{DUR_S/3600:.1f} h x {len(rec_groups)} ch ...", flush=True)
    noise = get_noise_levels(rec, return_in_uV=False)
    peaks = detect_peaks(rec, method="locally_exclusive", peak_sign="neg",
                         detect_threshold=DETECT_THRESH, radius_um=40.0,
                         noise_levels=noise, n_jobs=16, chunk_duration="1s", progress_bar=True)
    peak_s = peaks["sample_index"].astype(np.int64)
    peak_ch = peaks["channel_index"].astype(np.int64)
    peak_g = rec_groups[peak_ch].astype(np.int64)
    amp_mad = np.abs(peaks["amplitude"]) / noise[peak_ch]
    n_peaks = len(peak_s)
    print(f"\n{n_peaks:,} detected events ({n_peaks/(DUR_S):.0f}/s overall); "
          f"amp MAD: median={np.median(amp_mad):.1f} p90={np.percentile(amp_mad,90):.1f} "
          f"max={amp_mad.max():.1f}", flush=True)

    results = {}
    for label, dname in SORTINGS:
        p = OUT / dname
        if not p.exists():
            print(f"  (skip {label}: {dname} missing)", flush=True)
            continue
        srt = si.load(p)
        sbt, n_sorted = per_tetrode_sorted(srt, rec_groups)
        cm = claimed_mask(peak_s, peak_g, sbt, tol)
        results[label] = (cm, n_sorted, srt.get_num_units())
        print(f"\n=== {label}: {srt.get_num_units()} units, {n_sorted:,} sorted spikes ===", flush=True)
        print(f"  overall events claimed: {cm.mean()*100:.1f}%  "
              f"({cm.sum():,}/{n_peaks:,}); unclaimed {(~cm).sum():,}", flush=True)
        print(f"  {'amp band (MAD)':>16} {'#events':>11} {'%of events':>10} {'%claimed':>9} {'#unclaimed':>11}")
        for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
            m = (amp_mad >= lo) & (amp_mad < hi)
            if not m.any():
                continue
            band = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
            print(f"  {band:>16} {int(m.sum()):>11,} {m.mean()*100:>9.1f}% "
                  f"{cm[m].mean()*100:>8.1f}% {int((~cm[m]).sum()):>11,}", flush=True)

    # figure: claimed-fraction vs amplitude, + claimed/unclaimed amplitude histogram (reseed sorting)
    centers, fracs = [], {lab: [] for lab in results}
    for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
        m = (amp_mad >= lo) & (amp_mad < hi)
        if not m.any():
            continue
        centers.append(lo if not np.isfinite(hi) else 0.5 * (lo + hi))
        for lab, (cm, _, _) in results.items():
            fracs[lab].append(cm[m].mean() * 100)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for lab in results:
        ax[0].plot(centers, fracs[lab], marker="o", label=lab)
    ax[0].set_xlabel("event amplitude (MAD)")
    ax[0].set_ylabel("% of events claimed by a unit")
    ax[0].set_ylim(0, 100)
    ax[0].axhline(90, color="0.7", ls=":")
    ax[0].set_title("Coverage vs amplitude\n(low MAD = MUA hash; high MAD unclaimed = MISSED real units)")
    ax[0].legend(fontsize=8)
    if results:
        cm0 = next(iter(results.values()))[0]
        bins = np.linspace(5.5, min(40, amp_mad.max()), 60)
        ax[1].hist(amp_mad[cm0], bins=bins, color="#3b7dd8", alpha=0.8, label="claimed")
        ax[1].hist(amp_mad[~cm0], bins=bins, color="#d8743b", alpha=0.8, label="unclaimed")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("event amplitude (MAD)")
        ax[1].set_ylabel("# events (log)")
        ax[1].set_title(f"{list(results)[0]}: claimed vs unclaimed by amplitude")
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    pth = OUT / "spike_coverage.png"
    fig.savefig(pth, dpi=130)
    plt.close(fig)
    print(f"\nwrote {pth}", flush=True)

    np.savez(OUT / "spike_coverage.npz", peak_sample=peak_s, peak_group=peak_g, amp_mad=amp_mad,
             **{f"claimed_{i}": results[lab][0] for i, lab in enumerate(results)},
             labels=np.array(list(results)))
    print(f"wrote {OUT / 'spike_coverage.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

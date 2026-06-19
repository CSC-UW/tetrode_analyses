"""Spike-coverage of the CHUNK+MATCH deliverable (tracked_48h/analyzer_clustered.zarr), comparable to the
matching-pursuit coverage (script 64).

Reuses the SAME detected events from script 64 (`spike_coverage.npz`, in MP-binary frames) shifted to the
source-zarr absolute frame base, and matches them against the chunk+match global sorting -- so MP vs
chunk+match coverage is scored on IDENTICAL events, no second detect_peaks pass. The MP binary was
materialized from the zarr starting at START_S=2000 s, so binary_frame + START_S*FS = absolute frame, and
the chunk+match global sorting is in absolute (t0=0) frames.

NOTE analyzer_clustered.zarr is the CURATED (isolation-gated) deliverable -- only well-isolated tetrode
units survive -- so its coverage is expected to be LOWER than the un-curated MP bank but PURER. The
comparison is curated-chunk+match vs all-MP-bank; read accordingly.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/66_coverage_chunkmatch.py
"""
import pathlib

import numpy as np
import spikeinterface as si

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix")
OUT = ROOT / "track_eval/mp_long_s2000_d170000"
CHUNKMATCH = ROOT / "tracked_48h/analyzer_clustered.zarr"
FS = 30000.0
START_S = 2000.0  # MP-binary frame 0 == this many seconds into the source zarr
TOL_MS = 0.5
AMP_BINS = [5.5, 7, 9, 12, 16, 24, np.inf]


def per_tetrode_sorted(sorting):
    grp = np.asarray(sorting.get_property("group"))
    out = {}
    for i, u in enumerate(sorting.unit_ids):
        out.setdefault(int(grp[i]), []).append(sorting.get_unit_spike_train(u).astype(np.int64))
    return {g: np.sort(np.concatenate(v)) for g, v in out.items()}


def claimed_mask(peak_s, peak_g, sbt, tol):
    out = np.zeros(len(peak_s), dtype=bool)
    for g in np.unique(peak_g):
        st = sbt.get(int(g))
        if st is None or st.size == 0:
            continue
        sel = peak_g == g
        ps = peak_s[sel]
        j = np.searchsorted(st, ps)
        dprev = np.where(j > 0, ps - st[np.clip(j - 1, 0, st.size - 1)], tol + 1)
        dnext = np.where(j < st.size, st[np.clip(j, 0, st.size - 1)] - ps, tol + 1)
        out[np.flatnonzero(sel)] = np.minimum(dprev, dnext) <= tol
    return out


def main():
    d = np.load(OUT / "spike_coverage.npz")
    peak_abs = d["peak_sample"].astype(np.int64) + int(START_S * FS)  # -> source-zarr absolute frames
    peak_g = d["peak_group"].astype(np.int64)
    amp = d["amp_mad"]
    n = len(peak_abs)
    tol = int(TOL_MS * 1e-3 * FS)

    az = si.load_sorting_analyzer(CHUNKMATCH, load_extensions=False)
    srt = az.sorting
    sbt = per_tetrode_sorted(srt)
    n_sorted = int(sum(srt.get_unit_spike_train(u).size for u in srt.unit_ids))
    # the global sorting spans the full 48 h; peaks span [START_S, START_S+47h]. restrict the comparison
    # to events whose tetrode actually has sorted spikes in the peak span (all do); coverage is over the
    # peak span, identical to the MP measurement.
    cm = claimed_mask(peak_abs, peak_g, sbt, tol)
    print(f"chunk+match analyzer_clustered.zarr (CURATED): {srt.get_num_units()} units, "
          f"{n_sorted:,} sorted spikes", flush=True)
    print(f"overall events claimed: {cm.mean()*100:.1f}%  ({cm.sum():,}/{n:,}); "
          f"unclaimed {(~cm).sum():,}", flush=True)
    print(f"  {'amp band (MAD)':>16} {'#events':>11} {'%claimed':>9} {'#unclaimed':>11}")
    for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
        m = (amp >= lo) & (amp < hi)
        if not m.any():
            continue
        band = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {band:>16} {int(m.sum()):>11,} {cm[m].mean()*100:>8.1f}% {int((~cm[m]).sum()):>11,}",
              flush=True)

    # side-by-side vs the MP reseed deliverable (claimed_0 in the npz), same events
    mp = d["claimed_0"]
    print("\n  amplitude    MP-reseed%   chunk+match%   (same events)")
    for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
        m = (amp >= lo) & (amp < hi)
        if not m.any():
            continue
        band = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {band:>9}    {mp[m].mean()*100:>9.1f}%   {cm[m].mean()*100:>11.1f}%")
    print(f"\n  OVERALL      MP {mp.mean()*100:.1f}%   chunk+match {cm.mean()*100:.1f}%", flush=True)

    np.savez(OUT / "spike_coverage_chunkmatch.npz", claimed=cm, n_units=srt.get_num_units(),
             n_sorted=n_sorted)
    print(f"wrote {OUT / 'spike_coverage_chunkmatch.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

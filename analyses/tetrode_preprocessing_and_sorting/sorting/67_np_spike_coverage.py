"""Spike coverage of 48 h Neuropixels Kilosort sortings -- what fraction of detected spikes do units capture?

Same question as the tetrode measurement (scripts 64/66), on real NP probes, for context. Uses the peaks
the sorting pipeline ALREADY detected for drift correction (`peaks.npy`, full recording, in the sorting's
sample base) as the event reference -- no recording re-read. A peak is "claimed" if some unit fired within
+/-0.5 ms AND within R_UM depth of it (peak depth = peak_locations `y`; unit depth = unit_locations `y` for
the Kilosort sortings, or extremum-channel depth for a SortingAnalyzer -- both monopolar depth in um).
Stratified by amplitude in approximate MAD (peaks detected at detect_threshold MAD, read per recording).

Peaks are block-subsampled to ~N_SAMPLE (spread across the recording) for the coverage fraction; all
sorted spikes are kept as candidate claimers.

CNPIX12-Santiago imec0 uses the SortingAnalyzer (run separately, see header in the report). CNPIX4-Doppio
imec1 and CNPIX10-Charles imec0 store an older WaveformExtractor postpro, loaded via the WNE extractor.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/67_np_spike_coverage.py
"""
import pathlib

import numpy as np
import spikeinterface as si

BASE = pathlib.Path("/Volumes/npx_nfs/shared_s3/novel_objects_deprivation/full")
EXPERIMENT = "novel_objects_deprivation"
FS = 30000.0
TOL_MS = 0.5
R_UM = 75.0
N_SAMPLE = 5_000_000
N_BLOCKS = 200
AMP_BINS = [5, 7, 9, 12, 16, 24, np.inf]

TARGETS = [
    dict(label="CNPIX12-Santiago imec0", kind="analyzer",
         analyzer=BASE / "CNPIX12-Santiago/sorting.imec0/postpro_48h/si_sorting_analyzer",
         peaks_dir=BASE / "CNPIX12-Santiago/sorting.imec0/preprocessing"),
    dict(label="CNPIX4-Doppio imec1", kind="wne", subject="CNPIX4-Doppio", probe="imec1",
         peaks_dir=BASE / "CNPIX4-Doppio/sorting.imec1/motion_best_estimate",
         ul=BASE / "CNPIX4-Doppio/sorting.imec1/postpro_48h/si_output/unit_locations/unit_locations.npy"),
    dict(label="CNPIX10-Charles imec0", kind="wne", subject="CNPIX10-Charles", probe="imec0",
         peaks_dir=BASE / "CNPIX10-Charles/sorting.imec0/preprocessing",
         ul=BASE / "CNPIX10-Charles/sorting.imec0/postpro_48h/si_output/unit_locations/unit_locations.npy"),
]


def read_detect_threshold(peaks_dir, default=5.0):
    oy = pathlib.Path(peaks_dir) / "opts.yaml"
    if oy.exists():
        for ln in oy.read_text().splitlines():
            if "detect_threshold" in ln:
                try:
                    return float(ln.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return default


def load_sorting_depths(t):
    if t["kind"] == "analyzer":
        sa = si.load_sorting_analyzer(t["analyzer"], load_extensions=False)
        srt = sa.sorting
        ec = si.get_template_extremum_channel(sa, outputs="index")
        cl = sa.get_channel_locations()
        return srt, {u: float(cl[ec[u], 1]) for u in srt.unit_ids}
    import wisc_ecephys_tools as wet
    proj = wet.get_sglx_project("shared")
    ext = proj.get_kilosort_extractor(t["subject"], EXPERIMENT, t["probe"], alias="full",
                                      sorting="sorting", postprocessing="postpro_48h")
    ul = np.load(t["ul"])
    return ext, {u: float(ul[i, 1]) for i, u in enumerate(ext.unit_ids)}


def block_sample_peaks(peaks_dir, n_sample, n_blocks):
    pk = np.load(pathlib.Path(peaks_dir) / "peaks.npy", mmap_mode="r")
    loc = np.load(pathlib.Path(peaks_dir) / "peak_locations.npy", mmap_mode="r")
    n = len(pk)
    bs = max(1, min(n, n_sample) // n_blocks)
    offs = np.unique(np.linspace(0, max(0, n - bs), n_blocks).astype(np.int64))
    S, A, Y = [], [], []
    for o in offs:
        S.append(np.asarray(pk["sample_ind"][o:o + bs]).astype(np.int64))
        A.append(np.abs(np.asarray(pk["amplitude"][o:o + bs]).astype(np.float64)))
        Y.append(np.asarray(loc["y"][o:o + bs]).astype(np.float64))
    return np.concatenate(S), np.concatenate(A), np.concatenate(Y), n


def coverage(t):
    detect_thr = read_detect_threshold(t["peaks_dir"])
    sp, samp, sy, n_peaks = block_sample_peaks(t["peaks_dir"], N_SAMPLE, N_BLOCKS)
    scale = samp.min() / detect_thr
    amp_mad = samp / scale

    srt, depth = load_sorting_depths(t)
    times, depths = [], []
    for u in srt.unit_ids:
        tr = srt.get_unit_spike_train(u).astype(np.int64)
        times.append(tr)
        depths.append(np.full(tr.size, depth[u], dtype=np.float32))
    st = np.concatenate(times)
    sd = np.concatenate(depths)
    n_sorted = st.size
    sbin = np.floor(sd / R_UM).astype(np.int64)
    order = np.lexsort((st, sbin))
    st, sbin = st[order], sbin[order]
    bmin = int(sbin.min())
    edges = np.searchsorted(sbin, np.arange(bmin, int(sbin.max()) + 2))

    def bin_times(b):
        k = b - bmin
        return st[edges[k]:edges[k + 1]] if 0 <= k < len(edges) - 1 else np.empty(0, np.int64)

    pbin = np.floor(sy / R_UM).astype(np.int64)
    tol = int(TOL_MS * 1e-3 * FS)
    claimed = np.zeros(len(sp), bool)
    for b in np.unique(pbin):
        m = pbin == b
        pts = sp[m]
        cand = np.sort(np.concatenate([bin_times(b - 1), bin_times(b), bin_times(b + 1)]))
        if cand.size == 0:
            continue
        j = np.searchsorted(cand, pts)
        dprev = np.where(j > 0, pts - cand[np.clip(j - 1, 0, cand.size - 1)], tol + 1)
        dnext = np.where(j < cand.size, cand[np.clip(j, 0, cand.size - 1)] - pts, tol + 1)
        claimed[np.flatnonzero(m)] = np.minimum(dprev, dnext) <= tol

    print(f"\n===== {t['label']} =====", flush=True)
    print(f"  {srt.get_num_units()} units, {n_sorted:,} sorted spikes | {n_peaks:,} detected events "
          f"(thr {detect_thr:g} MAD, scale {scale:.1f}/MAD); sampled {len(sp):,} peaks", flush=True)
    print(f"  overall events claimed: {claimed.mean()*100:.1f}%", flush=True)
    print(f"  {'amp band (MAD)':>16} {'%of events':>11} {'%claimed':>9}", flush=True)
    for lo, hi in zip(AMP_BINS[:-1], AMP_BINS[1:]):
        mm = (amp_mad >= lo) & (amp_mad < hi)
        if not mm.any():
            continue
        band = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {band:>16} {mm.mean()*100:>10.1f}% {claimed[mm].mean()*100:>8.1f}%", flush=True)
    return (t["label"], srt.get_num_units(), n_sorted, n_peaks, float(claimed.mean()))


def main():
    import sys
    only = sys.argv[1:]  # optional label substrings to filter (e.g. "Doppio Charles")
    rows = []
    for t in TARGETS:
        if only and not any(o.lower() in t["label"].lower() for o in only):
            continue
        try:
            rows.append(coverage(t))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  {t['label']} FAILED: {repr(e)[:160]}", flush=True)
    print("\n" + "=" * 74)
    print(f"{'recording':>26} {'units':>6} {'sorted spk':>14} {'events':>14} {'claimed':>9}")
    for label, nu, ns, npk, frac in rows:
        print(f"{label:>26} {nu:>6} {ns:>14,} {npk:>14,} {frac*100:>8.1f}%")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

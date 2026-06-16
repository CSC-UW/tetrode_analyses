"""PoC stage 0: can matching pursuit re-detect a known MS5 sort from its own templates?

The tracking bottleneck is per-chunk isolation DROPOUT (clean units MS5 fails to isolate in some
chunk -> chain breaks; ~7% genuinely-missed, not matching-recoverable). A template-matching /
deconvolution sorter attacks this at the root: a unit detected by matching its template does NOT
need to be independently re-clustered each chunk, so it stops vanishing. SpikeInterface ships the
component -- `sortingcomponents.matching.find_spikes_from_templates(method="circus-omp")` (the
orthogonal-matching-pursuit peeler Lupin/SpyKING-CIRCUS use) -- and it's geometry-free: motion
correction is a SEPARATE upstream component we simply don't apply, and our tetrode probegroup
(tetrodes 300um apart, ~tens-of-um within) makes circus-omp's spatial sparsity isolate each tetrode.

This is the cheapest, riskiest-mechanics test FIRST (gate before scaling): on a short drift-stable
span, sort once with MS5 (the reference), build a Templates bank from that sort, run circus-omp over
the SAME span, and measure how well it reproduces the reference (compare_two_sorters; same frame base,
no shift). High agreement on well-isolated units => geometry-free matching pursuit works on our data
and is worth scaling to the dropout-recovery test (stage 1, a longer fragmenting span + carried/
re-estimated templates). Low => debug sparsity/geometry before investing further.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/51_match_pursuit_poc.py \
        [--start-s 36000] [--dur-s 1200] [--n-jobs 16]
"""
import argparse
import pathlib
import shutil
import time

import numpy as np
import spikeinterface as si
import spikeinterface.comparison as sc
from spikeinterface.core import ChannelSparsity, Templates
from spikeinterface.core.template_tools import get_dense_templates_array
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

from tetrode_analyses.tracking import Chunk, materialize_chunk, sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
ZARR = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
OUT = ROOT / "sortings_seed42_pcafix" / "track_eval" / "mp_poc"
FS = 30000.0
MS_BEFORE, MS_AFTER = 1.0, 2.0


def build_templates(ref_int, recording, n_jobs):
    """In-memory analyzer (group-sparse) -> (Templates bank, snr per unit, analyzer).

    Templates are built in RAW units (return_in_uV=False / is_in_uV=False) to MATCH the units
    circus-omp reads from the recording. The materialized binary carries gain_to_uV=0.195, so
    uV templates would be ~5x too small vs the raw traces -> OMP fits an inflated amplitude and
    cannot subtract the spike, re-detecting it (the over-detection seen with uV templates).
    """
    sparsity = ChannelSparsity.from_property(ref_int, recording, by_property="group")
    az = si.create_sorting_analyzer(ref_int, recording, format="memory", sparsity=sparsity, return_in_uV=False)
    az.compute({"random_spikes": {}, "waveforms": {"ms_before": MS_BEFORE, "ms_after": MS_AFTER},
                "templates": {}, "noise_levels": {}}, n_jobs=n_jobs)
    az.compute({"quality_metrics": {"metric_names": ["snr", "firing_rate"]}})  # snr = well-isolated proxy
    dense = get_dense_templates_array(az, return_in_uV=False)  # raw units, to match the traces circus-omp reads
    mask = az.sparsity.mask  # (n_units, n_channels) bool, per-tetrode (exactly 4 ch/unit)
    nbefore = az.get_extension("templates").nbefore
    # A Templates with a sparsity_mask expects the SPARSE array (n_units, n_samples, max_active):
    # pack each unit's template onto its tetrode's channels (channel order = mask True order).
    n_units, n_samp, _ = dense.shape
    n_act = int(mask.sum(axis=1).max())
    sparse_arr = np.zeros((n_units, n_samp, n_act), dtype=np.float32)
    for i in range(n_units):
        chans = np.flatnonzero(mask[i])
        sparse_arr[i, :, : chans.size] = dense[i][:, chans]
    templates = Templates(
        templates_array=sparse_arr, sampling_frequency=FS, nbefore=nbefore, is_in_uV=False,
        sparsity_mask=mask, channel_ids=np.asarray(recording.channel_ids),
        unit_ids=np.asarray(ref_int.unit_ids), probe=None, check_for_consistent_sparsity=True,
    )
    qm = az.get_extension("quality_metrics").get_data()
    return templates, qm


def mp_to_sorting(spikes, templates):
    """circus-omp structured-array output -> NumpySorting (template index -> unit id)."""
    names = spikes.dtype.names
    unit_field = "cluster_index" if "cluster_index" in names else "unit_index"
    samples = spikes["sample_index"].astype(np.int64)
    labels = np.asarray(templates.unit_ids)[spikes[unit_field].astype(np.int64)]
    return si.NumpySorting.from_samples_and_labels(
        [samples], [labels], sampling_frequency=FS, unit_ids=np.asarray(templates.unit_ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=36000.0, help="span start (s); default drift-stable")
    ap.add_argument("--dur-s", type=float, default=1200.0, help="span duration (s); default 20 min")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--only-default", action="store_true", help="run only the default circus-omp setting (fast diagnostic)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT / "ref_sort", ignore_errors=True)  # idempotent: MS5 won't overwrite by_group/*

    chunk = Chunk(index=0, start_frame=int(args.start_s * FS),
                  end_frame=int((args.start_s + args.dur_s) * FS), fs=FS)
    print(f"span [{args.start_s:.0f}, {args.start_s + args.dur_s:.0f}) s = {args.dur_s/60:.0f} min", flush=True)

    t0 = time.perf_counter()
    rec = materialize_chunk(ZARR, chunk, OUT / "binary", cmr="global", n_jobs=96)
    print(f"[{time.perf_counter()-t0:.0f}s] materialized: {rec.get_num_channels()}ch, "
          f"probe={rec.get_probegroup() is not None}", flush=True)

    t1 = time.perf_counter()
    ref = sort_chunk(rec, OUT / "ref_sort")
    ref_int = to_int_numpy_sorting(ref)
    print(f"[{time.perf_counter()-t1:.0f}s] MS5 reference: {ref_int.get_num_units()} units", flush=True)

    t2 = time.perf_counter()
    templates, qm = build_templates(ref_int, rec, args.n_jobs)
    print(f"[{time.perf_counter()-t2:.0f}s] built {len(templates.unit_ids)} templates "
          f"(nbefore={templates.nbefore}, nsamples={templates.num_samples})", flush=True)

    snr = qm["snr"].to_numpy()
    nsp = np.array([len(ref_int.get_unit_spike_train(u)) for u in ref_int.unit_ids])
    well = (snr >= 5.0) & (nsp >= 50)  # high-SNR proxy for well-isolated
    n_ref = int(nsp.sum())
    inf = float("inf")

    # Default circus-omp params are tuned for high-density probes; on 4-channel tetrode templates
    # they over-detect (match noise). Sweep the acceptance knobs -- amplitude band (fitted scaling
    # relative to the template; reject weak/merged) and omp_min_sps (min normalized scalar product
    # to propose a match) -- to find a setting that suppresses false positives. Templates are built
    # once; only the cheap matching step re-runs per setting.
    SETTINGS = [
        ("default", {}),
        ("amp>=0.8", {"amplitudes": [0.8, inf]}),
        ("amp0.8-1.5", {"amplitudes": [0.8, 1.5]}),
        ("amp0.9-1.5", {"amplitudes": [0.9, 1.5]}),
        ("sps0.5", {"omp_min_sps": 0.5}),
        ("amp0.8-1.5+sps0.5", {"amplitudes": [0.8, 1.5], "omp_min_sps": 0.5}),
    ]
    if args.only_default:
        SETTINGS = [("default", {})]
    print(f"\nref total spikes={n_ref}  well-isolated units={int(well.sum())}/{len(nsp)}")
    print(f"{'setting':22s} {'n_spk':>8s} {'ratio':>6s} {'well_med':>9s} {'>=0.5':>6s} {'>=0.8':>6s} {'t(s)':>5s}")
    best_default = None
    for name, mk in SETTINGS:
        t = time.perf_counter()
        spikes = find_spikes_from_templates(
            rec, templates, method="circus-omp", method_kwargs=mk,
            job_kwargs={"n_jobs": args.n_jobs, "chunk_duration": "1s", "progress_bar": False})
        mp = mp_to_sorting(spikes, templates)
        cmp = sc.compare_two_sorters(ref_int, mp, sorting1_name="ms5_ref", sorting2_name="circus_omp")
        agree = cmp.agreement_scores.values
        best = np.nanmax(agree, axis=1) if agree.size else np.zeros(len(nsp))
        wb = best[well]
        print(f"{name:22s} {len(spikes):8d} {len(spikes)/max(n_ref,1):6.1f} {np.median(wb):9.3f} "
              f"{np.mean(wb>=0.5):6.2f} {np.mean(wb>=0.8):6.2f} {time.perf_counter()-t:5.0f}", flush=True)
        if name == "default":
            best_default = best
            # Diagnose the 4.4x over-detection: is it duplicate assignment (same physical spike
            # matched to several near-identical oversplit templates) or genuinely spurious spikes?
            amp = np.asarray(spikes["amplitude"], dtype=float)
            pct = np.percentile(amp, [1, 10, 50, 90, 99])
            # collapse detections within 0.5 ms ACROSS all units -> count unique physical events
            s = np.sort(np.asarray(spikes["sample_index"], dtype=np.int64))
            uniq = 1 + int((np.diff(s) > int(0.5e-3 * FS)).sum()) if s.size else 0
            print(f"  [diag] fitted-amplitude pct[1,10,50,90,99]={np.round(pct,2)}", flush=True)
            print(f"  [diag] circus spikes={len(s)} -> unique events (>0.5ms apart)={uniq} "
                  f"(ref={n_ref}); collapse ratio={len(s)/max(uniq,1):.1f}x", flush=True)
            # LABEL-AGNOSTIC detection recall: do circus spikes land on ref spikes at all
            # (ignoring unit identity)? high recall + low agreement => pure label-splitting across
            # redundant oversplit templates (dedup fixes it); low recall => genuine misses.
            tol = int(0.5e-3 * FS)
            ref_well = np.sort(np.concatenate(
                [ref_int.get_unit_spike_train(u) for u, w in zip(ref_int.unit_ids, well) if w]).astype(np.int64))
            j = np.searchsorted(s, ref_well)
            dprev = np.where(j > 0, ref_well - s[np.clip(j - 1, 0, len(s) - 1)], tol + 1)
            dnext = np.where(j < len(s), s[np.clip(j, 0, len(s) - 1)] - ref_well, tol + 1)
            recall = float(np.mean(np.minimum(dprev, dnext) <= tol))
            print(f"  [diag] label-agnostic detection recall on well-isolated ref spikes "
                  f"(<=0.5ms): {recall:.3f}", flush=True)
    np.savez(OUT / "poc_agreement.npz", best_default=best_default, snr=snr, n_spikes=nsp, well=well)
    print(f"\nwrote {OUT/'poc_agreement.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

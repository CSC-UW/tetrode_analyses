"""Stage 2 (long): does carry-forward matching pursuit track confident units across HOURS of drift?

Seed a confident template bank from an MS5 sort of the FIRST window only (realistic: you seed at the
start, not from a whole-span sort), then detect those units across a multi-hour span window-by-window,
two ways: FIXED (initial templates throughout) vs REESTIMATE (re-derive present units' templates each
window, carrying old templates forward for absent units). The per-window continuity TREND is the
signal: if fixed fades over time while reestimate stays flat, template-matching is tracking drift
(geometry-free) where chunk+match would fragment. Cleanliness (ISI<1ms) guards against noise tracks.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/54_match_pursuit_long_drift.py \
        [--start-s 56000] [--dur-s 36000] [--window-s 1800] [--seed-window-s 1800] [--n-jobs 16]
"""
import argparse
import pathlib
import shutil

import numpy as np

from _mp_common import build_templates_object, dedup_sorting, materialize_span, windowed_carry_forward
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
FS = 30000.0
AMP = {"amplitudes": [0.8, float("inf")]}


def isi_viol_frac(train, refr_ms=1.0):
    if len(train) < 2:
        return np.nan
    return float(np.mean(np.diff(np.sort(train)) / FS * 1000.0 < refr_ms))


def report(name, asm, counts, conf_ids):
    present = counts >= 20  # >=20 spikes/window
    per_win = present.mean(axis=1)             # fraction of units present, per window (the TREND)
    per_unit = present.mean(axis=0)            # fraction of windows present, per unit
    isi = np.array([isi_viol_frac(asm.get_unit_spike_train(u)) for u in conf_ids])
    print(f"{name:10s} | unit continuity: median={np.median(per_unit):.2f} "
          f"frac_full(>=0.95)={np.mean(per_unit>=0.95):.2f} | ISI<1ms med={np.nanmedian(isi):.4f}", flush=True)
    print(f"{name:10s} | per-window present-fraction trend: "
          f"{np.array2string(per_win, precision=2, max_line_width=200)}", flush=True)
    return per_win, per_unit, isi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=56000.0)
    ap.add_argument("--dur-s", type=float, default=36000.0)        # 10 h
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--seed-window-s", type=float, default=1800.0)  # first 30 min -> seed bank
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--method", choices=["circus-omp", "wobble"], default="circus-omp",
                    help="matcher. circus-omp uses amplitudes=[0.8,inf]; wobble sets its per-window "
                    "threshold from --wobble-factor and optionally adds the --shape-gate-r cosine gate.")
    ap.add_argument("--wobble-factor", type=float, default=None,
                    help="wobble admit threshold = factor x median ||t||^2 of the per-window bank "
                    "(permissive e.g. 0.45 + --shape-gate-r = cosine arm; binding ~0.55-0.70 = adaptive arm).")
    ap.add_argument("--shape-gate-r", type=float, default=None,
                    help="scale-invariant cosine acceptance gate: keep spikes with cos(snippet, assigned "
                    "template) >= this. Applies to any --method.")
    ap.add_argument("--min-spikes", type=int, default=100)
    ap.add_argument("--min-spikes-reestimate", type=int, default=None,
                    help="template-reliability floor for per-window re-estimation; default = --min-spikes "
                    "(tie to the seed-confidence bar: a re-estimated template needs as many spikes as the "
                    "initial one, else carry the prior). Set explicitly to DECOUPLE when lowering --min-spikes.")
    ap.add_argument("--skip-fixed", action="store_true",
                    help="run only the reestimate pass (the deliverable; halves compute on long spans)")
    ap.add_argument("--skip-reestimate", action="store_true",
                    help="run only the fixed pass (reuses the cached binary; for fixed-vs-reestimate comparison)")
    ap.add_argument("--tag", default="", help="suffix for output files (avoid clobbering a prior run's npz/dirs)")
    ap.add_argument("--dedup-cosine", type=float, default=None,
                    help="if set, merge within-tetrode near-identical seed units (template cosine >= this) "
                    "BEFORE seeding the bank, removing MS5 oversplit redundancy (e.g. 0.9)")
    ap.add_argument("--reestimate-min-cos", type=float, default=None,
                    help="per-window re-estimation STEP-CAP (e.g. 0.8): reject a window's re-estimated "
                    "template if its 4-ch shift-cosine to the current template < this -> blocks the "
                    "identity-swap capture (a track walking onto a louder same-tetrode neighbor)")
    args = ap.parse_args()
    if args.method == "wobble" and args.wobble_factor is None:
        ap.error("--method wobble requires --wobble-factor")
    mk = None if args.method == "wobble" else AMP  # wobble derives its per-window threshold from --wobble-factor
    reest_floor = args.min_spikes_reestimate if args.min_spikes_reestimate is not None else args.min_spikes
    out = ROOT / f"mp_long_s{int(args.start_s)}_d{int(args.dur_s)}"
    out.mkdir(parents=True, exist_ok=True)

    rec = materialize_span(out, args.start_s, args.dur_s)
    n_win = int(np.ceil(args.dur_s / args.window_s))
    print(f"span [{args.start_s:.0f},{args.start_s+args.dur_s:.0f})s  {n_win} windows of {args.window_s:.0f}s", flush=True)
    print(f"matcher={args.method}"
          + (f" (wobble_factor={args.wobble_factor}, shape_gate_r={args.shape_gate_r})"
             if args.method == "wobble" else " (amplitudes=[0.8,inf])"), flush=True)

    # seed bank from the FIRST window's MS5 sort (confident, well-isolated units)
    seed_frames = int(args.seed_window_s * FS)
    win0 = rec.frame_slice(0, seed_frames)
    win0.reset_times()
    shutil.rmtree(out / f"seed_sort{args.tag}", ignore_errors=True)
    ref0 = to_int_numpy_sorting(sort_chunk(win0, out / f"seed_sort{args.tag}"))
    _, az = build_templates_object(ref0, win0, with_snr=True, n_jobs=args.n_jobs)
    snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
    nsp = np.array([len(ref0.get_unit_spike_train(u)) for u in ref0.unit_ids])
    well = (snr >= 5.0) & (nsp >= args.min_spikes)
    conf_ids = np.asarray(ref0.unit_ids)[well]
    print(f"seed: {ref0.get_num_units()} units in window0 -> {len(conf_ids)} confident (snr>=5 & >={args.min_spikes} spk)", flush=True)
    seed_sorting = ref0.select_units(conf_ids)
    if args.dedup_cosine:
        n_before = seed_sorting.get_num_units()
        seed_sorting = dedup_sorting(seed_sorting, win0, cosine_min=args.dedup_cosine)
        print(f"dedup (within-tetrode cosine>={args.dedup_cosine}): {n_before} -> {seed_sorting.get_num_units()} seed units", flush=True)
    init_templates, _ = build_templates_object(seed_sorting, win0, with_snr=False, n_jobs=args.n_jobs)

    track_ids = np.asarray(seed_sorting.unit_ids)  # deduped seed ids (== conf_ids if no dedup)
    print(f"\n{'mode':10s} | drift tracking across {n_win} windows", flush=True)
    saved = {"conf_ids": track_ids}
    if not args.skip_fixed:
        asm_fx, cnt_fx = windowed_carry_forward(rec, init_templates, window_s=args.window_s,
                                                method=args.method, method_kwargs=mk,
                                                shape_gate_r=args.shape_gate_r, wobble_factor=args.wobble_factor,
                                                n_jobs=args.n_jobs, reestimate=False)
        pw_fx, pu_fx, isi_fx = report("fixed", asm_fx, cnt_fx, track_ids)
        saved.update(counts_fixed=cnt_fx, perwin_fixed=pw_fx, perunit_fixed=pu_fx, isi_fixed=isi_fx)
    if not args.skip_reestimate:
        asm_re, cnt_re = windowed_carry_forward(rec, init_templates, window_s=args.window_s,
                                                method=args.method, method_kwargs=mk,
                                                shape_gate_r=args.shape_gate_r, wobble_factor=args.wobble_factor,
                                                n_jobs=args.n_jobs, reestimate=True,
                                                min_spikes_reestimate=reest_floor,
                                                reestimate_min_cos=args.reestimate_min_cos)
        pw_re, pu_re, isi_re = report("reestimate", asm_re, cnt_re, track_ids)
        saved.update(counts_reest=cnt_re, perwin_reest=pw_re, perunit_reest=pu_re, isi_reest=isi_re)
        # persist the reestimate assembled sorting (the deliverable: continuous tracks)
        asm_re.save(folder=out / f"assembled_reestimate{args.tag}", overwrite=True)
    np.savez(out / f"long_drift{args.tag}.npz", **saved)
    print(f"\nwrote {out/('long_drift'+args.tag+'.npz')}\nDONE", flush=True)


if __name__ == "__main__":
    main()

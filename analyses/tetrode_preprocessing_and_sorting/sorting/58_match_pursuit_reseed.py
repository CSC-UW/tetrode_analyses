"""Stage 3 (prototype): carry-forward matching pursuit with PERIODIC RE-SEEDING.

Seeds a confident template bank from the first window's MS5 sort (like script 54), then carries it
forward window-by-window WITH re-estimation AND periodic re-seeding: every --reseed-every-windows
windows, re-sort that window with MS5 and add any confident cluster that doesn't match an existing
bank unit on its tetrode (shift-cos < --reseed-add-cos) as a NEW tracked unit. This gives a
late-appearing / ramping neuron its own template so it claims its own spikes instead of being captured
by a same-tetrode neighbour (the IDENTITY-SWAP root-cause fix; see MATCHING_PURSUIT_FINDINGS). Also
tracks units that first appear after the seed window.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/58_match_pursuit_reseed.py \
        --start-s 2000 --dur-s 170000 --dedup-cosine 0.95 --reseed-every-windows 12 --tag _reseed
    # smoke: add --max-windows 5 --reseed-every-windows 2  (first 5 windows on the cached binary)
    # wobble primary (cosine-gate arm): --method wobble --wobble-factor 0.45 --shape-gate-r 0.6 --tag _wobble
"""
import argparse
import pathlib
import shutil

import numpy as np

from _mp_common import (
    build_templates_object,
    dedup_sorting,
    materialize_span,
    windowed_carry_forward_reseed,
)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
FS = 30000.0
AMP = {"amplitudes": [0.8, float("inf")]}


def isi_viol_frac(train, refr_ms=1.0):
    if len(train) < 2:
        return np.nan
    return float(np.mean(np.diff(np.sort(train)) / FS * 1000.0 < refr_ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=2000.0)
    ap.add_argument("--dur-s", type=float, default=170000.0)
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--seed-window-s", type=float, default=1800.0)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--method", choices=["circus-omp", "wobble"], default="circus-omp",
                    help="matcher. circus-omp uses amplitudes=[0.8,inf] (scale-invariant ratio gate); "
                    "wobble sets its per-window threshold from --wobble-factor and optionally adds the "
                    "--shape-gate-r cosine acceptance gate.")
    ap.add_argument("--wobble-factor", type=float, default=None,
                    help="wobble admit threshold = factor x median ||t||^2 of the per-window bank. A "
                    "permissive value (e.g. 0.45) + --shape-gate-r realises the cosine-gate arm; a "
                    "binding value (~0.55-0.70) is the adaptive-||t||^2 arm.")
    ap.add_argument("--shape-gate-r", type=float, default=None,
                    help="scale-invariant cosine acceptance gate: keep spikes with cos(snippet, assigned "
                    "template) >= this. Applies to any --method.")
    ap.add_argument("--min-spikes", type=int, default=100)
    ap.add_argument("--min-spikes-reestimate", type=int, default=None,
                    help="template-reliability floor for per-window re-estimation; default = --min-spikes "
                    "(tie to the seed/re-seed admission bar). Set explicitly to DECOUPLE when lowering --min-spikes.")
    ap.add_argument("--dedup-cosine", type=float, default=0.95)
    ap.add_argument("--reseed-every-windows", type=int, default=12)
    ap.add_argument("--reseed-add-cos", type=float, default=0.8)
    ap.add_argument("--max-windows", type=int, default=None, help="smoke: process only first N windows")
    ap.add_argument("--tag", default="_reseed")
    args = ap.parse_args()
    if args.method == "wobble" and args.wobble_factor is None:
        ap.error("--method wobble requires --wobble-factor (e.g. 0.45 for the cosine-gate arm with "
                 "--shape-gate-r, or ~0.55-0.70 for the adaptive-||t||^2 arm)")
    reest_floor = args.min_spikes_reestimate if args.min_spikes_reestimate is not None else args.min_spikes
    out = ROOT / f"mp_long_s{int(args.start_s)}_d{int(args.dur_s)}"
    out.mkdir(parents=True, exist_ok=True)

    rec = materialize_span(out, args.start_s, args.dur_s)
    n_win = int(np.ceil(args.dur_s / args.window_s))
    print(f"span [{args.start_s:.0f},{args.start_s + args.dur_s:.0f})s  {n_win} windows of {args.window_s:.0f}s; "
          f"reseed every {args.reseed_every_windows} windows (add cos<{args.reseed_add_cos})", flush=True)
    print(f"matcher={args.method}"
          + (f" (wobble_factor={args.wobble_factor}, shape_gate_r={args.shape_gate_r})"
             if args.method == "wobble" else " (amplitudes=[0.8,inf])"), flush=True)

    # seed bank from the FIRST window's MS5 sort (same recipe as script 54)
    seed_frames = int(args.seed_window_s * FS)
    win0 = rec.frame_slice(0, seed_frames)
    win0.reset_times()
    shutil.rmtree(out / f"seed_sort{args.tag}", ignore_errors=True)
    ref0 = to_int_numpy_sorting(sort_chunk(win0, out / f"seed_sort{args.tag}"))
    _, az = build_templates_object(ref0, win0, with_snr=True, n_jobs=args.n_jobs)
    snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
    nsp = np.array([len(ref0.get_unit_spike_train(u)) for u in ref0.unit_ids])
    conf_ids = np.asarray(ref0.unit_ids)[(snr >= 5.0) & (nsp >= args.min_spikes)]
    seed_sorting = ref0.select_units(conf_ids)
    if args.dedup_cosine:
        seed_sorting = dedup_sorting(seed_sorting, win0, cosine_min=args.dedup_cosine)
    init_templates, _ = build_templates_object(seed_sorting, win0, with_snr=False, n_jobs=args.n_jobs)
    n_seed = seed_sorting.get_num_units()
    print(f"seed: {len(conf_ids)} confident -> {n_seed} after dedup{args.dedup_cosine}", flush=True)

    shutil.rmtree(out / f"reseed_sorts{args.tag}", ignore_errors=True)
    mk = None if args.method == "wobble" else AMP  # wobble derives its per-window threshold from --wobble-factor
    asm, counts, births, births_cos = windowed_carry_forward_reseed(
        rec, init_templates, window_s=args.window_s, method=args.method, method_kwargs=mk,
        shape_gate_r=args.shape_gate_r, wobble_factor=args.wobble_factor, n_jobs=args.n_jobs,
        min_spikes_reestimate=reest_floor, reseed_min_spikes=args.min_spikes,
        reseed_every_windows=args.reseed_every_windows, reseed_add_cos=args.reseed_add_cos,
        reseed_dir=out / f"reseed_sorts{args.tag}", max_windows=args.max_windows)

    all_ids = np.asarray(sorted(counts))
    cnt_mat = np.stack([counts[u] for u in all_ids])  # (n_units, n_windows)
    present = cnt_mat >= 20
    per_unit = present.mean(axis=1)
    isi = np.array([isi_viol_frac(asm.get_unit_spike_train(u)) for u in all_ids])
    print(f"\nRESEED: {n_seed} seed -> {len(all_ids)} total units ({len(births)} added by re-seeding)", flush=True)
    print(f"  continuity median per-unit present-frac={np.median(per_unit):.2f} | "
          f"ISI<1ms median={np.nanmedian(isi):.4f} | units >1% refractory={int(np.nansum(isi > 0.01))}", flush=True)
    if births:
        bw = np.array(sorted(births.values()))
        print(f"  re-seeded births at windows: {sorted(set(bw.tolist()))}", flush=True)
        bcos = np.array([births_cos[u] for u in births])
        print(f"  born-unit at-birth cos-to-bank: median={np.median(bcos):.2f} "
              f"max={np.max(bcos):.2f} | borderline(>=0.6)={int(np.sum(bcos >= 0.6))}/{len(bcos)}",
              flush=True)

    bids = list(births)
    asm.save(folder=out / f"assembled_reseed{args.tag}", overwrite=True)
    np.savez(out / f"reseed{args.tag}.npz", all_ids=all_ids, counts=cnt_mat, isi=isi,
             birth_ids=np.array(bids), birth_win=np.array([births[u] for u in bids]),
             birth_cos=np.array([births_cos[u] for u in bids]),
             seed_ids=np.asarray(seed_sorting.unit_ids))
    print(f"\nwrote {out / ('assembled_reseed' + args.tag)} and {out / ('reseed' + args.tag + '.npz')}\nDONE", flush=True)


if __name__ == "__main__":
    main()

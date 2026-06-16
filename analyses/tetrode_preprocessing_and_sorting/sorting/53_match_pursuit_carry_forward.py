"""Stage 2: carry-forward matching pursuit across drift (the tracking deliverable).

Build a confident (well-isolated) template bank from an MS5 sort of the span, then detect those units
across the span window-by-window with circus-omp, two ways: REESTIMATE (re-derive each present unit's
template each window, carrying old templates forward for absent units -> tracks drift) vs FIXED
(initial templates throughout). Compare per-unit CONTINUITY (fraction of windows the unit is present)
and train CLEANLINESS (ISI<1ms). On a drifting span, reestimate should keep units continuous where
fixed templates fade -- demonstrating template-matching tracks drift without geometry, recovering the
per-chunk dropout that fragments chunk+match.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/53_match_pursuit_carry_forward.py \
        [--start-s 36000] [--dur-s 1800] [--window-s 600] [--n-jobs 16]
"""
import argparse
import pathlib

import numpy as np

from _mp_common import build_templates_object, prepare_span, windowed_carry_forward

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_stage2")
FS = 30000.0
AMP = {"amplitudes": [0.8, float("inf")]}


def isi_viol_frac(train, refr_ms=1.0):
    if len(train) < 2:
        return np.nan
    isi_ms = np.diff(np.sort(train)) / FS * 1000.0
    return float(np.mean(isi_ms < refr_ms))


def summarize(name, asm, counts, conf_ids):
    present = counts >= 20  # >=20 spikes in a window = "present"
    cont = present.mean(axis=0)  # per unit: fraction of windows present
    isi = np.array([isi_viol_frac(asm.get_unit_spike_train(u)) for u in conf_ids])
    full = np.mean(cont >= 0.99)  # fraction of units present in ~all windows
    print(f"{name:10s} | continuity: median={np.median(cont):.2f} mean={np.mean(cont):.2f} "
          f"frac_full(>=0.99)={full:.2f} | ISI<1ms median={np.nanmedian(isi):.4f} "
          f"p90={np.nanpercentile(isi,90):.4f}", flush=True)
    return cont, isi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=36000.0)
    ap.add_argument("--dur-s", type=float, default=1800.0)
    ap.add_argument("--window-s", type=float, default=600.0)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--min-spikes", type=int, default=100, help="min ref spikes for a confident unit")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec, ref = prepare_span(OUT, args.start_s, args.dur_s)
    n_win = int(np.ceil(args.dur_s / args.window_s))
    print(f"span [{args.start_s:.0f},{args.start_s+args.dur_s:.0f})s  {n_win} windows of {args.window_s:.0f}s  "
          f"ref units={ref.get_num_units()}", flush=True)

    # confident (well-isolated) template bank
    _, az = build_templates_object(ref, rec, with_snr=True, n_jobs=args.n_jobs)
    snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
    nsp = np.array([len(ref.get_unit_spike_train(u)) for u in ref.unit_ids])
    well = (snr >= 5.0) & (nsp >= args.min_spikes)
    conf_ids = np.asarray(ref.unit_ids)[well]
    print(f"confident units: {len(conf_ids)}/{ref.get_num_units()} (snr>=5 & >={args.min_spikes} spk)", flush=True)
    conf = ref.select_units(conf_ids)
    init_templates, _ = build_templates_object(conf, rec, with_snr=False, n_jobs=args.n_jobs)

    print(f"\n{'mode':10s} | continuity + cleanliness across {n_win} windows", flush=True)
    asm_fx, cnt_fx = windowed_carry_forward(rec, init_templates, window_s=args.window_s,
                                            method_kwargs=AMP, n_jobs=args.n_jobs, reestimate=False)
    summarize("fixed", asm_fx, cnt_fx, conf_ids)
    asm_re, cnt_re = windowed_carry_forward(rec, init_templates, window_s=args.window_s,
                                            method_kwargs=AMP, n_jobs=args.n_jobs, reestimate=True)
    cont_re, isi_re = summarize("reestimate", asm_re, cnt_re, conf_ids)

    np.savez(OUT / "carry_forward.npz", counts_fixed=cnt_fx, counts_reest=cnt_re,
             conf_ids=conf_ids, cont_reest=cont_re, isi_reest=isi_re)
    print(f"\nwrote {OUT/'carry_forward.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

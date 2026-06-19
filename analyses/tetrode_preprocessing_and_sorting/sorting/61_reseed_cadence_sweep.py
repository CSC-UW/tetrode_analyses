"""Stage 3 sweep: re-seed CADENCE sweep -- new-unit YIELD vs duplicate-TWIN curves.

Re-seeding's one demonstrated value is new-unit yield (the same-seed A/B showed it is swap-NEUTRAL;
see MATCHING_PURSUIT_FINDINGS). A FINER cadence picks up late/ramping neurons sooner, but also gives
more chances to re-add a drifting unit as a spurious "new" track (a drift-split TWIN). This script finds
where yield saturates and twins start climbing.

ONE seed bank is built once (identical recipe to script 58) and SHARED across all cadences, so cadence is
the only variable (same-seed philosophy, as in the A/B -- removes the seed-luck confound). For each cadence
(re-seed every N windows) it runs windowed_carry_forward_reseed and, for the units that cadence ADDS,
reports:

  yield        number of re-seeded units
  birth_cos    each born unit's max 4-ch template cosine to the existing SAME-TETRODE bank AT BIRTH -- the
               exact quantity the add-cos gate thresholds on, matched-in-time. LOW = a confidently distinct
               neuron; near the 0.8 gate = a borderline duplicate of a neighbour (drift-split twin risk).
  final_twin   post-hoc: the born unit's FULL-SPAN template max-cos to any same-tetrode unit that existed
               BEFORE it (catches duplicates that converge AFTER birth). >= --twin-cos (0.9) = a twin.
  clean        ISI<1ms violation fraction < 1%
  present      fraction of windows the born unit is present (>=20 spk); ~0 = stillborn (spurious birth)

The promising-cadence read: genuine yield (low birth_cos, clean, present) should SATURATE while the
borderline/twin count CLIMBS as cadence gets finer. The knee is the cadence to adopt.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/61_reseed_cadence_sweep.py \
        --start-s 2000 --dur-s 170000 --dedup-cosine 0.95 --cadences 12,6,3
    # smoke: --max-windows 6 --cadences 4,2   (first 6 windows on the cached binary)
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
from tetrode_analyses.tracking import cosine_from_templates, sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
FS = 30000.0
AMP = {"amplitudes": [0.8, float("inf")]}


def isi_viol_frac(train, refr_ms=1.0):
    if len(train) < 2:
        return np.nan
    return float(np.mean(np.diff(np.sort(train)) / FS * 1000.0 < refr_ms))


def fullspan_t4(asm, rec, *, n_jobs, min_spikes_template=50):
    """{unit_id: (T, 4) full-span template, group} for asm units with enough spikes to template.

    Span-averaged templates (random_spikes caps at 500/unit) on each unit's own tetrode channels.
    Units below min_spikes_template (e.g. stillborn births) are omitted. Returns (t4, ugroup)."""
    groups_asm = np.asarray(asm.get_property("group"))
    gmap = {int(u): int(groups_asm[i]) for i, u in enumerate(asm.unit_ids)}
    keep = [u for u in asm.unit_ids if len(asm.get_unit_spike_train(u)) >= min_spikes_template]
    asm_keep = asm.select_units(keep)
    asm_keep.set_property("group", np.array([gmap[int(u)] for u in asm_keep.unit_ids]))
    templates, _ = build_templates_object(asm_keep, rec, with_snr=False, n_jobs=n_jobs)
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)  # (n, T, n_chan)
    rec_groups = np.asarray(rec.get_property("group"))
    t4, ug = {}, {}
    for i, u in enumerate([int(x) for x in templates.unit_ids]):
        gch = np.flatnonzero(rec_groups == gmap[u])
        t4[u] = dense[i][:, gch]
        ug[u] = gmap[u]
    return t4, ug


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=2000.0)
    ap.add_argument("--dur-s", type=float, default=170000.0)
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--seed-window-s", type=float, default=1800.0)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--min-spikes", type=int, default=100)
    ap.add_argument("--min-spikes-reestimate", type=int, default=None,
                    help="re-estimation template-reliability floor; default = --min-spikes (tie to the bar)")
    ap.add_argument("--dedup-cosine", type=float, default=0.95)
    ap.add_argument("--cadences", default="12,6,3",
                    help="comma list of reseed-every-windows values (12,6,3 windows = 6h,3h,1.5h at 1800s)")
    ap.add_argument("--reseed-add-cos", type=float, default=0.8)
    ap.add_argument("--twin-cos", type=float, default=0.9,
                    help="full-span template cosine to a pre-existing same-tetrode unit that flags a TWIN")
    ap.add_argument("--borderline-cos", type=float, default=0.6,
                    help="at-birth cos-to-bank at/above which a born unit is a BORDERLINE duplicate")
    ap.add_argument("--max-windows", type=int, default=None, help="smoke: process only first N windows")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    reest_floor = args.min_spikes_reestimate if args.min_spikes_reestimate is not None else args.min_spikes
    cadences = [int(x) for x in args.cadences.split(",") if x.strip()]
    out = ROOT / f"mp_long_s{int(args.start_s)}_d{int(args.dur_s)}"
    out.mkdir(parents=True, exist_ok=True)

    rec = materialize_span(out, args.start_s, args.dur_s)
    n_win = int(np.ceil(args.dur_s / args.window_s))
    print(f"span [{args.start_s:.0f},{args.start_s + args.dur_s:.0f})s  {n_win} windows of {args.window_s:.0f}s; "
          f"cadence sweep over every-{cadences} windows ({[args.window_s * c / 3600 for c in cadences]} h)",
          flush=True)

    # ---- ONE shared seed bank (identical recipe to script 58; shared across cadences) ----
    seed_frames = int(args.seed_window_s * FS)
    win0 = rec.frame_slice(0, seed_frames)
    win0.reset_times()
    shutil.rmtree(out / f"seed_sort_sweep{args.tag}", ignore_errors=True)
    ref0 = to_int_numpy_sorting(sort_chunk(win0, out / f"seed_sort_sweep{args.tag}"))
    _, az = build_templates_object(ref0, win0, with_snr=True, n_jobs=args.n_jobs)
    snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
    nsp = np.array([len(ref0.get_unit_spike_train(u)) for u in ref0.unit_ids])
    conf_ids = np.asarray(ref0.unit_ids)[(snr >= 5.0) & (nsp >= args.min_spikes)]
    seed_sorting = ref0.select_units(conf_ids)
    if args.dedup_cosine:
        seed_sorting = dedup_sorting(seed_sorting, win0, cosine_min=args.dedup_cosine)
    init_templates, _ = build_templates_object(seed_sorting, win0, with_snr=False, n_jobs=args.n_jobs)
    n_seed = seed_sorting.get_num_units()
    seed_id_set = {int(u) for u in init_templates.unit_ids}
    print(f"seed (shared): {len(conf_ids)} confident -> {n_seed} after dedup{args.dedup_cosine}", flush=True)

    rows = []
    saved = {"cadences": np.array(cadences), "n_seed": n_seed,
             "window_s": args.window_s, "twin_cos": args.twin_cos, "borderline_cos": args.borderline_cos}
    for N in cadences:
        ctag = f"_c{N}{args.tag}"
        shutil.rmtree(out / f"reseed_sorts{ctag}", ignore_errors=True)
        print(f"\n=== cadence: re-seed every {N} windows ({args.window_s * N / 3600:.1f} h) ===", flush=True)
        asm, counts, births, births_cos = windowed_carry_forward_reseed(
            rec, init_templates, window_s=args.window_s, method_kwargs=AMP, n_jobs=args.n_jobs,
            min_spikes_reestimate=reest_floor, reseed_min_spikes=args.min_spikes,
            reseed_every_windows=N, reseed_add_cos=args.reseed_add_cos,
            reseed_dir=out / f"reseed_sorts{ctag}", max_windows=args.max_windows)
        asm.save(folder=out / f"assembled_reseed{ctag}", overwrite=True)

        born = list(births)
        bcos = np.array([births_cos[u] for u in born]) if born else np.zeros(0)
        present = np.array([float((counts[u] >= 20).mean()) for u in born]) if born else np.zeros(0)
        isi = np.array([isi_viol_frac(asm.get_unit_spike_train(u)) for u in born]) if born else np.zeros(0)

        # post-hoc twin check: born unit's full-span template vs same-tetrode units that existed BEFORE it
        t4, ug = fullspan_t4(asm, rec, n_jobs=args.n_jobs)
        final_cos = np.full(len(born), np.nan)
        for i, b in enumerate(born):
            if b not in t4:
                continue  # stillborn / too few spikes to template
            g, wb = ug[b], births[b]
            pre = [u for u in t4 if ug[u] == g and u != b
                   and (u in seed_id_set or (u in births and births[u] < wb))]
            if pre:
                final_cos[i] = max(cosine_from_templates(t4[b], t4[u]) for u in pre)

        n_borderline = int(np.sum(bcos >= args.borderline_cos)) if born else 0
        n_distinct = int(np.sum(bcos < args.borderline_cos)) if born else 0
        n_twin = int(np.nansum(final_cos >= args.twin_cos))
        n_clean = int(np.nansum(isi < 0.01))
        n_present = int(np.sum(present >= 0.25)) if born else 0  # tracked in >=1/4 of windows
        # GENUINE = distinct at birth AND not a post-hoc twin AND clean AND persistently present
        genuine = np.array(
            [(bcos[i] < args.borderline_cos) and (not (final_cos[i] >= args.twin_cos))
             and (isi[i] < 0.01 if np.isfinite(isi[i]) else False) and (present[i] >= 0.25)
             for i in range(len(born))]) if born else np.zeros(0, bool)
        n_genuine = int(np.sum(genuine))

        rows.append((N, args.window_s * N / 3600, len(born), n_genuine, n_distinct, n_borderline,
                     n_twin, n_clean, n_present,
                     float(np.median(bcos)) if born else np.nan,
                     float(np.max(bcos)) if born else np.nan,
                     float(np.median(present)) if born else np.nan))
        saved[f"born_ids_c{N}"] = np.array(born)
        saved[f"birth_win_c{N}"] = np.array([births[u] for u in born])
        saved[f"birth_cos_c{N}"] = bcos
        saved[f"final_cos_c{N}"] = final_cos
        saved[f"isi_c{N}"] = isi
        saved[f"present_c{N}"] = present
        print(f"  -> {len(born)} born | genuine={n_genuine} distinct(birth<{args.borderline_cos})={n_distinct} "
              f"borderline={n_borderline} | final-twin(>= {args.twin_cos})={n_twin} | "
              f"clean-ISI={n_clean} present(>=0.25)={n_present}", flush=True)

    # ---- summary table ----
    hdr = (f"{'every':>6} {'hours':>6} {'born':>5} {'genuine':>8} {'distinct':>9} {'border':>7} "
           f"{'twin':>5} {'clean':>6} {'present':>8} {'bcos_med':>9} {'bcos_max':>9} {'pres_med':>9}")
    print("\n" + "=" * len(hdr))
    print("RESEED CADENCE SWEEP  (yield saturates + twins climb -> adopt the knee)")
    print("=" * len(hdr))
    print(hdr)
    for r in rows:
        print(f"{r[0]:>6d} {r[1]:>6.1f} {r[2]:>5d} {r[3]:>8d} {r[4]:>9d} {r[5]:>7d} "
              f"{r[6]:>5d} {r[7]:>6d} {r[8]:>8d} {r[9]:>9.2f} {r[10]:>9.2f} {r[11]:>9.2f}")

    np.savez(out / f"reseed_cadence_sweep{args.tag}.npz", **saved)
    print(f"\nwrote {out / ('reseed_cadence_sweep' + args.tag + '.npz')}\nDONE", flush=True)


if __name__ == "__main__":
    main()

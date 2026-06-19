"""Same-seed A/B: does periodic re-seeding PREVENT the identity-swap (clean attribution)?

Seeds ONE deterministic template bank (build_templates_object now seeds random_spikes, so the dedup'd
seed is reproducible), then runs the carry-forward TWO ways from that identical bank over the same
window range:
  arm A = no re-seeding (windowed_carry_forward, reestimate)
  arm B = periodic re-seeding (windowed_carry_forward_reseed)
The ONLY difference is re-seeding on/off, so any swap-rate difference is cleanly attributable to it.
Mechanism predicts: arm A leaves late-appearing units UNSEEDED (capture bait) -> more swaps; arm B
gives them their own tracks -> fewer. Metric: per-SEED-unit early[1,13]h <-> late[24,30]h template
self-cosine (low = identity changed); same seed ids in both arms.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/59_ab_reseed_swap.py [--max-windows 60]
"""
# ruff: noqa: E702  (compact one-line frame_slice setup, intentional)
import argparse
import pathlib
import shutil

import numpy as np
from spikeinterface.core.template_tools import get_dense_templates_array

from _mp_common import (
    build_templates_object,
    dedup_sorting,
    materialize_span,
    windowed_carry_forward,
    windowed_carry_forward_reseed,
)
from tetrode_analyses.tracking import cosine_from_templates, sort_chunk, to_int_numpy_sorting

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval")
FS = 30000.0
AMP = {"amplitudes": [0.8, float("inf")]}
EARLY = (1.0, 13.0)


def self_cos_per_unit(asm, rec, unit_ids, late):
    """early<->late shift-cos for each of unit_ids present (>=30 spk) in BOTH bins."""
    def bin_t(lo, hi):
        a0, b0 = int(lo * 3600 * FS), int(hi * 3600 * FS)
        s = asm.frame_slice(a0, b0); rr = rec.frame_slice(a0, b0); rr.reset_times()
        pres = [u for u in unit_ids if len(s.get_unit_spike_train(u)) >= 30]
        if not pres:
            return {}
        _, az = build_templates_object(s.select_units(pres), rr, with_snr=False, n_jobs=16)
        dn = get_dense_templates_array(az, return_in_uV=False); mk = az.sparsity.mask
        return {int(u): dn[i][:, np.flatnonzero(mk[i])] for i, u in enumerate([int(x) for x in az.unit_ids])}
    E, L = bin_t(*EARLY), bin_t(*late)
    return {u: cosine_from_templates(E[u], L[u], max_shift_samples=10) for u in unit_ids if u in E and u in L}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-s", type=float, default=2000.0)
    ap.add_argument("--dur-s", type=float, default=170000.0)
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--max-windows", type=int, default=60)  # 30 h; covers the u20 capture (done by ~24 h)
    ap.add_argument("--reseed-every-windows", type=int, default=12)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    out = ROOT / f"mp_long_s{int(args.start_s)}_d{int(args.dur_s)}"
    late = (args.max_windows * args.window_s / 3600.0 - 6.0, args.max_windows * args.window_s / 3600.0)

    rec = materialize_span(out, args.start_s, args.dur_s)
    # ONE deterministic seed bank (random_spikes seeded -> reproducible dedup)
    win0 = rec.frame_slice(0, int(args.window_s * FS)); win0.reset_times()
    shutil.rmtree(out / "seed_sort_ab", ignore_errors=True)
    ref0 = to_int_numpy_sorting(sort_chunk(win0, out / "seed_sort_ab"))
    _, az = build_templates_object(ref0, win0, with_snr=True, n_jobs=args.n_jobs)
    snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
    nsp = np.array([len(ref0.get_unit_spike_train(u)) for u in ref0.unit_ids])
    conf = np.asarray(ref0.unit_ids)[(snr >= 5.0) & (nsp >= 100)]
    seed_sorting = dedup_sorting(ref0.select_units(conf), win0, cosine_min=0.95)
    init_templates, _ = build_templates_object(seed_sorting, win0, with_snr=False, n_jobs=args.n_jobs)
    seed_ids = [int(u) for u in seed_sorting.unit_ids]
    print(f"deterministic seed: {len(conf)} confident -> {len(seed_ids)} after dedup0.95; "
          f"A/B over {args.max_windows} windows, late bin {late[0]:.0f}-{late[1]:.0f}h", flush=True)

    print("\n=== arm A: NO re-seeding ===", flush=True)
    asm_a, _ = windowed_carry_forward(rec, init_templates, window_s=args.window_s, method_kwargs=AMP,
                                      n_jobs=args.n_jobs, reestimate=True, max_windows=args.max_windows)
    asm_a.save(folder=out / "assembled_ab_noreseed", overwrite=True)

    print("\n=== arm B: WITH re-seeding ===", flush=True)
    shutil.rmtree(out / "reseed_sorts_ab", ignore_errors=True)
    asm_b, _, births = windowed_carry_forward_reseed(
        rec, init_templates, window_s=args.window_s, method_kwargs=AMP, n_jobs=args.n_jobs,
        reseed_every_windows=args.reseed_every_windows, reseed_dir=out / "reseed_sorts_ab",
        max_windows=args.max_windows)

    sa = self_cos_per_unit(asm_a, rec, seed_ids, late)
    sb = self_cos_per_unit(asm_b, rec, seed_ids, late)
    common = sorted(set(sa) & set(sb))
    av = np.array([sa[u] for u in common]); bv = np.array([sb[u] for u in common])
    print(f"\n=== RESULT (per-seed-unit early<->late self-cos; {len(common)} units in both arms) ===", flush=True)
    print(f"  arm A (no reseed): median={np.median(av):.3f}  <0.7: {(av < 0.7).sum()}  <0.5: {(av < 0.5).sum()}", flush=True)
    print(f"  arm B (reseed)   : median={np.median(bv):.3f}  <0.7: {(bv < 0.7).sum()}  <0.5: {(bv < 0.5).sum()}", flush=True)
    print(f"  arm B added {len(births)} new units (reseeded)", flush=True)
    print("  seed units that SWAP in A (self-cos<0.7) and their B value (B>A => reseed rescued):", flush=True)
    for u in sorted(common, key=lambda u: sa[u]):
        if sa[u] < 0.7:
            print(f"    u{u}: A={sa[u]:.3f} -> B={sb[u]:.3f}  {'RESCUED' if sb[u] - sa[u] > 0.15 else ''}", flush=True)
    np.savez(out / "ab_reseed_swap.npz", seed_ids=np.array(seed_ids), common=np.array(common),
             self_cos_A=av, self_cos_B=bv, birth_ids=np.array(list(births)))
    print(f"\nwrote {out / 'ab_reseed_swap.npz'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

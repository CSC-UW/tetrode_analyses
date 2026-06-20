"""Part C / E4 -- per-tetrode MUA bucket: capture real-but-unsortable events as is_mua-flagged pseudo-units.

After SUA (+ SUA residual-capture), the events still unclaimed and >= a MAD floor are EITHER neural-but-
unsortable (MUA) OR noise. The plan's foundational neural-vs-noise signal is waveform SHAPE: an event is
MUA if its best cosine to ANY same-tetrode SUA template is in [theta, SUA_TAU) -- it matches a known neural
shape but not cleanly enough to assign to a single unit; below theta it matches nothing -> noise (dropped).
theta comes from E1 (script 91). Surviving events are pooled into ONE MUA pseudo-unit per tetrode (preserves
spatial localization), coincident spikes (<0.3 ms) collapsed, persisted with an ``is_mua`` unit property
(False for SUA, True for MUA). BombCell shape labels on the MUA units are reported as a secondary check.

Order matters (plan s4): SUA -> SUA-residual-capture -> MUA, so the bucket never eats recoverable SUA;
here the SUA base is the merge-first candidate, and events with best-cos >= SUA_TAU are left for Part A
(NOT put in MUA). State (ON/OFF, sleep) is NOT used -- it validates the MUA product downstream only (E5).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/92_mua_pass.py \
        [--sua assembled_mergefirst] [--theta 0.5] [--mad-floor 7] [--windows-h 5 26 40]
"""
import argparse
import json
import pathlib

import numpy as np
import spikeinterface as si

from _assignment_eval import best_template_for_events, window_bank
from _mp_common import FS, materialize_span
from _scoreboard import coverage_by_band

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
SUA_TAU = 0.8
COINCIDENCE_MS = 0.3


def load_theta(cli_theta):
    if cli_theta is not None:
        return cli_theta
    p = OUT / "neural_noise_calibration.json"
    if p.exists():
        t = json.loads(p.read_text()).get("theta_suggested")
        if t is not None and np.isfinite(t):
            return float(t)
    return 0.5  # provisional fallback if E1 not yet run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sua", default="assembled_mergefirst")
    ap.add_argument("--theta", type=float, default=None, help="MUA cosine floor (default from E1 json)")
    ap.add_argument("--mad-floor", type=float, default=7.0)
    ap.add_argument("--windows-h", nargs="+", type=float, default=[5.0, 26.0, 40.0])
    ap.add_argument("--out-name", default="assembled_mergefirst_mua")
    args = ap.parse_args()
    theta = load_theta(args.theta)

    rec = materialize_span(OUT, START_S, DUR_S)
    nfr = rec.get_num_frames()
    sua = si.load(OUT / args.sua)
    grp_of = {int(u): int(g) for u, g in zip(sua.unit_ids, np.asarray(sua.get_property("group")))}
    print(f"SUA base {args.sua}: {sua.get_num_units()} units; theta={theta}, MAD floor={args.mad_floor}, "
          f"windows {args.windows_h}h", flush=True)

    z = np.load(OUT / "spike_coverage.npz", mmap_mode="r")
    peak_s = np.asarray(z["peak_sample"])
    peak_g = np.asarray(z["peak_group"])
    amp_mad = np.asarray(z["amp_mad"])
    # whole-recording claimed mask vs the SUA base (pooled per tetrode, +/-0.5 ms)
    _, claimed = coverage_by_band(sua, peak_s, peak_g, amp_mad)
    mua_input = (~claimed) & (amp_mad >= args.mad_floor)
    print(f"  unclaimed >= {args.mad_floor} MAD events (MUA input): {int(mua_input.sum()):,}", flush=True)

    mua_frames = {}      # tetrode group -> list of absolute MUA frames
    n_mua = n_noise = n_sua_recoverable = n_total = 0
    for h in args.windows_h:
        a = int(h * 3600 * FS)
        b = min(a + int(WIN_S * FS), nfr)
        if a >= nfr:
            continue
        win, bank, _ = window_bank(rec, sua, a, b, n_jobs=N_JOBS)
        if bank is None:
            continue
        lo, hi = np.searchsorted(peak_s, [a, b])
        sel = mua_input[lo:hi]
        es = (peak_s[lo:hi][sel] - a).astype(np.int64)
        eg = peak_g[lo:hi][sel].astype(np.int64)
        if es.size == 0:
            continue
        best_cos, _, valid = best_template_for_events(win, bank, es, eg)
        n_total += int(valid.sum())
        is_mua = valid & (best_cos >= theta) & (best_cos < SUA_TAU)
        is_sua_rec = valid & (best_cos >= SUA_TAU)   # leave for Part A, not MUA
        is_noise = valid & (best_cos < theta)
        n_mua += int(is_mua.sum())
        n_sua_recoverable += int(is_sua_rec.sum())
        n_noise += int(is_noise.sum())
        for g in np.unique(eg[is_mua]):
            mua_frames.setdefault(int(g), []).append(es[is_mua & (eg == g)] + a)
        print(f"  window {h}h: {es.size:,} unclaimed -> MUA {int(is_mua.sum()):,}, "
              f"noise {int(is_noise.sum()):,}, sua-recoverable {int(is_sua_rec.sum()):,}", flush=True)

    # assemble: SUA units (is_mua=False) + one MUA pseudo-unit per tetrode (is_mua=True)
    tol = int(COINCIDENCE_MS * 1e-3 * FS)
    trains, groups, ismua = {}, [], []
    for u in sua.unit_ids:
        trains[int(u)] = np.sort(sua.get_unit_spike_train(u).astype(np.int64))
        groups.append(grp_of[int(u)])
        ismua.append(False)
    next_id = max(int(u) for u in sua.unit_ids) + 1
    mua_counts = {}
    for g, chunks in sorted(mua_frames.items()):
        allspk = np.sort(np.concatenate(chunks))
        if allspk.size and tol > 0:
            allspk = allspk[np.concatenate([[True], np.diff(allspk) > tol])]
        trains[next_id] = allspk
        groups.append(g)
        ismua.append(True)
        mua_counts[g] = int(allspk.size)
        next_id += 1
    merged = si.NumpySorting.from_unit_dict([trains], sampling_frequency=FS)
    merged.set_property("group", np.asarray(groups))
    merged.set_property("is_mua", np.asarray(ismua))
    out_dir = OUT / args.out_name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    merged.save(folder=out_dir)

    print(f"\n  classified {n_total:,} unclaimed events -> MUA {n_mua:,} ({100*n_mua/max(n_total,1):.1f}%), "
          f"noise {n_noise:,} ({100*n_noise/max(n_total,1):.1f}%), "
          f"sua-recoverable {n_sua_recoverable:,} ({100*n_sua_recoverable/max(n_total,1):.1f}%)", flush=True)
    print(f"  built {len(mua_counts)} per-tetrode MUA pseudo-units (is_mua=True); "
          f"events/tetrode median {int(np.median(list(mua_counts.values()))) if mua_counts else 0}", flush=True)
    print(f"  saved {sua.get_num_units()} SUA + {len(mua_counts)} MUA units -> {out_dir}", flush=True)

    (OUT / "mua_pass.json").write_text(json.dumps({
        "sua_base": args.sua, "theta": theta, "mad_floor": args.mad_floor, "windows_h": args.windows_h,
        "n_classified": n_total, "n_mua": n_mua, "n_noise": n_noise, "n_sua_recoverable": n_sua_recoverable,
        "n_mua_units": len(mua_counts), "mua_events_per_tetrode": mua_counts,
    }, indent=2))
    print(f"\nwrote {OUT / 'mua_pass.json'}\nDONE", flush=True)


if __name__ == "__main__":
    main()

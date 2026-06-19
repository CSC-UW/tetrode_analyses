"""Persist matching-pursuit curation columns onto a carry-forward / re-seed analyzer.

Script 56 builds the analyzer BEFORE the identity check (55) exists, so `identity_min_cos` and the
track-extent columns are not on the analyzer. This is the SPOT step that writes them (previously a
manual one-off, which is why analyzer_tracks_rs.zarr shipped without them). Sets, via
`set_sorting_property(..., save=True)` (the only persistence that survives reload -- direct zarr writes
to sorting/properties/ are ignored, see TRACKING_FINDINGS gotcha):

  identity_min_cos  : from identity_check<...>.npz (NaN for units not present at all 5 sample points,
                      e.g. late-born re-seeded units -> "not assessed")
  n_windows         : windows the unit is present (>=20 spk), from the counts npz
  track_hours       : n_windows * window_s / 3600
  is_reseeded       : 1 for units added by periodic re-seeding (reseed npz only), else 0
  reseed_birth_h    : hour a re-seeded unit was added (NaN for seed units; reseed npz only)

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/60_persist_mp_curation_columns.py \
        --analyzer analyzer_tracks_rs.zarr --npz reseed_rs.npz --identity identity_check_rs.npz
"""
import argparse
import pathlib

import numpy as np
import spikeinterface as si

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_long_s2000_d170000")
WIN_S = 1800.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analyzer", required=True, help="analyzer zarr folder name under the run dir")
    ap.add_argument("--npz", required=True, help="counts npz (long_drift<tag>.npz or reseed<tag>.npz)")
    ap.add_argument("--identity", default=None, help="identity_check<tag>.npz (optional; sets identity_min_cos)")
    args = ap.parse_args()

    az = si.load_sorting_analyzer(OUT / args.analyzer)
    uids = [int(u) for u in az.sorting.unit_ids]
    pos = {u: i for i, u in enumerate(uids)}
    n = len(uids)
    d = np.load(OUT / args.npz)

    # counts -> (n_units, n_windows) aligned to analyzer unit order
    if "counts" in d.files:  # reseed npz: counts (n_units, nwin), all_ids
        all_ids = [int(u) for u in d["all_ids"]]
        cnt = {u: d["counts"][i] for i, u in enumerate(all_ids)}
    else:                    # long_drift npz: counts_reest (nwin, n_units), conf_ids
        conf = [int(u) for u in d["conf_ids"]]
        cm = d["counts_reest"]
        cnt = {u: cm[:, i] for i, u in enumerate(conf)}

    n_windows = np.zeros(n, dtype=np.int64)
    for u in uids:
        if u in cnt:
            n_windows[pos[u]] = int((cnt[u] >= 20).sum())
    track_hours = n_windows * WIN_S / 3600.0

    az.set_sorting_property("n_windows", n_windows, save=True)
    az.set_sorting_property("track_hours", track_hours.astype(np.float32), save=True)
    msg = ["n_windows", "track_hours"]

    if args.identity:
        ic = np.load(OUT / args.identity)
        icmap = {int(u): float(c) for u, c in zip(ic["uids"], ic["min_cos"])}
        idc = np.array([icmap.get(u, np.nan) for u in uids], dtype=np.float32)
        az.set_sorting_property("identity_min_cos", idc, save=True)
        msg.append(f"identity_min_cos ({np.isfinite(idc).sum()}/{n} assessed)")

    if "birth_ids" in d.files:
        births = {int(b): int(w) for b, w in zip(d["birth_ids"], d["birth_win"])}
        is_re = np.array([1 if u in births else 0 for u in uids], dtype=np.int64)
        birth_h = np.array([births[u] * WIN_S / 3600.0 if u in births else np.nan for u in uids], dtype=np.float32)
        az.set_sorting_property("is_reseeded", is_re, save=True)
        az.set_sorting_property("reseed_birth_h", birth_h, save=True)
        msg.append(f"is_reseeded ({int(is_re.sum())} reseeded), reseed_birth_h")

    print(f"persisted onto {args.analyzer}: {', '.join(msg)}", flush=True)
    print("properties now:", sorted(az.sorting.get_property_keys()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

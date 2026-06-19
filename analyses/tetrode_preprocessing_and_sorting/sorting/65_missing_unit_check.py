"""Are the unclaimed large spikes (script 64) REAL units the MP bank is MISSING, or dropout of banked units?

MS5-resort a few windows spanning the recording (early / sleep-dep / late). For each WELL-ISOLATED MS5
unit (snr>=5, >=100 spikes, ISI<1.5 ms violation <1%) ask two things against the MP deliverable
(assembled_reseed_rs):
  claimed_frac = fraction of THIS unit's spikes that an MP unit claims (within +/-0.5 ms, same tetrode)
  cos_to_bank  = max 4-ch template cosine to any MP bank unit on its tetrode
A clean, isolated MS5 unit with LOW claimed_frac AND LOW cos_to_bank is a unit the bank never had (its
large spikes are exactly the unclaimed-large events). A unit with LOW claimed_frac but HIGH cos_to_bank
would instead be a banked unit the matcher dropped (detection dropout). This separates the two causes.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/65_missing_unit_check.py
"""
import pathlib
import shutil

import numpy as np

from _mp_common import build_templates_object, materialize_span
from tetrode_analyses.tracking import cosine_from_templates, sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
WINDOW_STARTS_H = [5.0, 26.0, 40.0]  # binary-hour: early / sleep-dep / late
TOL_MS = 0.5
CLAIMED_LO = 0.5   # below this = the bank does not capture this unit
COS_BANK = 0.7     # above this = plausibly the same waveform as a bank unit (drift), not "missing"


def per_tetrode_sorted(sorting):
    grp = np.asarray(sorting.get_property("group"))
    out = {}
    for i, u in enumerate(sorting.unit_ids):
        out.setdefault(int(grp[i]), []).append(sorting.get_unit_spike_train(u).astype(np.int64))
    return {g: np.sort(np.concatenate(v)) for g, v in out.items()}


def claimed_fraction(train, mp_sorted, tol):
    if train.size == 0:
        return np.nan
    if mp_sorted.size == 0:
        return 0.0
    j = np.searchsorted(mp_sorted, train)
    dprev = np.where(j > 0, train - mp_sorted[np.clip(j - 1, 0, mp_sorted.size - 1)], tol + 1)
    dnext = np.where(j < mp_sorted.size, mp_sorted[np.clip(j, 0, mp_sorted.size - 1)] - train, tol + 1)
    return float(np.mean(np.minimum(dprev, dnext) <= tol))


def isi_viol(train, refr_ms=1.5):
    if train.size < 2:
        return np.nan
    return float(np.mean(np.diff(np.sort(train)) / FS * 1000.0 < refr_ms))


def unit_t4(templates, rec_groups):
    dense = np.asarray(templates.get_dense_templates(), dtype=np.float32)
    mask = templates.sparsity.mask
    out, grp = {}, {}
    for i, u in enumerate([int(x) for x in templates.unit_ids]):
        g = int(rec_groups[np.flatnonzero(mask[i])[0]])
        out[u] = dense[i][:, np.flatnonzero(rec_groups == g)]
        grp[u] = g
    return out, grp


def main():
    import spikeinterface as si
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    tol = int(TOL_MS * 1e-3 * FS)

    # MP deliverable: bank templates (full-span) + per-tetrode spike trains
    asm = si.load(OUT / "assembled_reseed_rs")
    mp_sbt = per_tetrode_sorted(asm)  # binary frames
    gmap = {int(u): int(g) for u, g in zip(asm.unit_ids, np.asarray(asm.get_property("group")))}
    keep = [u for u in asm.unit_ids if asm.get_unit_spike_train(u).size >= 50]
    ak = asm.select_units(keep)
    ak.set_property("group", np.array([gmap[int(u)] for u in ak.unit_ids]))
    bank_t, _ = build_templates_object(ak, rec, with_snr=False, n_jobs=16)
    bank_t4, bank_grp = unit_t4(bank_t, rec_groups)
    bank_by_tet = {}
    for u, g in bank_grp.items():
        bank_by_tet.setdefault(g, []).append(bank_t4[u])
    print(f"MP bank: {len(bank_t4)} units across {len(bank_by_tet)} tetrodes\n", flush=True)

    cov = np.load(OUT / "spike_coverage.npz")
    peak_s = cov["peak_sample"].astype(np.int64)
    peak_g = cov["peak_group"].astype(np.int64)
    amp = cov["amp_mad"]
    claimed0 = cov["claimed_0"]  # MP-reseed claimed mask (binary frames)

    rows = []
    for h in WINDOW_STARTS_H:
        a = int(h * 3600 * FS)
        b = min(a + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a, b)
        win.reset_times()
        sdir = OUT / f"missing_chk_w{int(h)}h"
        shutil.rmtree(sdir, ignore_errors=True)
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        templ, az = build_templates_object(ms5, win, with_snr=True, n_jobs=16)
        snr = az.get_extension("quality_metrics").get_data()["snr"].to_numpy()
        t4, grp = unit_t4(templ, rec_groups)
        ms5_ids = [int(u) for u in ms5.unit_ids]
        # MP spikes in this window, per tetrode, in window-local frames
        mp_local = {g: (v[(v >= a) & (v < b)] - a) for g, v in mp_sbt.items()}

        n_iso = n_missing = 0
        win_missing, claimed_list, iso_train = [], [], {}
        for i, u in enumerate(ms5_ids):
            tr = ms5.get_unit_spike_train(u).astype(np.int64)
            if snr[i] < 5.0 or tr.size < 100 or (isi_viol(tr) or 1.0) >= 0.01:
                continue
            n_iso += 1
            g = grp[u]
            iso_train.setdefault(g, []).append(tr)  # pooled well-isolated clean train
            cfrac = claimed_fraction(tr, mp_local.get(g, np.empty(0, np.int64)), tol)
            cos = max((cosine_from_templates(t4[u], bt) for bt in bank_by_tet.get(g, [])), default=-1.0)
            claimed_list.append(cfrac)
            if cfrac < CLAIMED_LO and cos < COS_BANK:
                n_missing += 1
                win_missing.append((u, g, snr[i], tr.size, cfrac, cos))
        iso_train = {g: np.sort(np.concatenate(v)) for g, v in iso_train.items()}
        cl = np.array(claimed_list) if claimed_list else np.zeros(0)

        # decompose the unclaimed-LARGE (>=12 MAD) events in this window: are they isolable by a clean
        # MS5 unit (=> within-unit DROPOUT of a banked unit) or in no clean unit (=> overlap/MUA)?
        sel = (peak_s >= a) & (peak_s < b) & (~claimed0) & (amp >= 12)
        ev_s, ev_g = peak_s[sel] - a, peak_g[sel]
        isol = np.zeros(len(ev_s), bool)
        for g in np.unique(ev_g):
            st = iso_train.get(int(g))
            if st is None or st.size == 0:
                continue
            m = ev_g == g
            es = ev_s[m]
            j = np.searchsorted(st, es)
            dp = np.where(j > 0, es - st[np.clip(j - 1, 0, st.size - 1)], tol + 1)
            dn = np.where(j < st.size, st[np.clip(j, 0, st.size - 1)] - es, tol + 1)
            isol[np.flatnonzero(m)] = np.minimum(dp, dn) <= tol

        print(f"=== window @ {h:.0f} h: {ms5.get_num_units()} MS5 units, {n_iso} well-isolated, "
              f"{n_missing} missing-from-bank ===", flush=True)
        if cl.size:
            print(f"  well-isolated unit claimed% by MP bank: median={np.median(cl)*100:.0f}%  "
                  f"[<50%:{int((cl<0.5).sum())}, 50-80%:{int(((cl>=0.5)&(cl<0.8)).sum())}, "
                  f"80-95%:{int(((cl>=0.8)&(cl<0.95)).sum())}, >95%:{int((cl>=0.95).sum())}]", flush=True)
        if len(ev_s):
            print(f"  unclaimed-large(>=12 MAD) events={len(ev_s):,}: isolable by a clean MS5 unit "
                  f"(banked-unit DROPOUT) {isol.mean()*100:.0f}%, NOT isolable (overlap/MUA) "
                  f"{(~isol).mean()*100:.0f}%", flush=True)
        for u, g, s, ns, cf, cos in sorted(win_missing, key=lambda x: -x[2]):
            print(f"    MISSING u{u} tet{g} snr{s:.1f} n{ns} claimed{cf*100:.0f}% cos2bank{cos:.2f}",
                  flush=True)
        rows.append((h, n_iso, n_missing, float(np.median(cl)) if cl.size else np.nan,
                     float(isol.mean()) if len(ev_s) else np.nan, int(len(ev_s))))

    print("\n" + "=" * 78)
    print("SUMMARY -- decomposition of the MP large-spike coverage gap")
    print("=" * 78)
    print(f"  {'win':>5} {'iso':>4} {'missing':>8} {'median claimed%':>16} "
          f"{'unclaimed-large isolable%':>26}")
    for h, niso, nmiss, medcl, isolf, nev in rows:
        print(f"  {h:>4.0f}h {niso:>4} {nmiss:>8} {medcl*100:>15.0f}% {isolf*100:>25.0f}%  (n={nev:,})")
    print("\nReading: 0 missing + high median claimed% => the bank is NOT missing clean isolated units; "
          "the unclaimed-large gap = within-unit DROPOUT (events isolable, inside a banked unit) plus "
          "NON-isolable overlap/MUA -- NOT absent units.", flush=True)


if __name__ == "__main__":
    main()

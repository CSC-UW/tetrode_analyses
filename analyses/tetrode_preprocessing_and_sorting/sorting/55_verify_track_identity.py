"""Verify identity stability of the 48h carry-forward tracks (guard against silent unit swaps).

The 48h deliverable tracks ~95% of confident units across the recording, clean + continuous. But
matching pursuit on 4-channel templates can confuse similar units, so a clean+continuous track could
silently SWAP identity to a template-similar neighbor. This samples each tracked unit's template at 5
time points across the 47h (from the saved assembled sorting) and reports consecutive-template cosine:
smooth high cosine = stable identity (drift is gradual); a sudden drop between adjacent points flags a
possible swap. Coarse (5 points) but cheap -- a gross sanity check on the headline.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/55_verify_track_identity.py
"""
import argparse
import pathlib

import numpy as np
import spikeinterface as si
from spikeinterface.core.template_tools import get_dense_templates_array

from _mp_common import build_templates_object
from tetrode_analyses.tracking import cosine_from_templates

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_long_s2000_d170000")
FS = 30000.0
WIN = int(900 * FS)  # +/-15 min around each sample point


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="suffix selecting a variant run (e.g. _dedup09); "
                    "reads assembled_reestimate<tag>, writes identity_check<tag>.npz")
    tag = ap.parse_args().tag
    rec = si.load(OUT / "binary")
    asm = si.load(OUT / f"assembled_reestimate{tag}")
    total = rec.get_num_frames()
    centers = np.linspace(WIN, total - WIN, 5).astype(int)  # 5 time points
    print(f"recording {total/FS/3600:.1f}h, sampling templates at {[round(c/FS/3600,1) for c in centers]} h", flush=True)

    # per (timepoint, unit) dense template, restricted to units with spikes at that point
    dense_by_t, mask_ref, present_by_t = [], None, []
    for c in centers:
        a, b = c - WIN, c + WIN
        r = rec.frame_slice(a, b); r.reset_times()
        s = asm.frame_slice(a, b)
        present = np.array([len(s.get_unit_spike_train(u)) >= 20 for u in asm.unit_ids])
        present_by_t.append(present)
        keep = np.asarray(asm.unit_ids)[present]
        t, az = build_templates_object(s.select_units(keep), r, with_snr=False)
        dense = get_dense_templates_array(az, return_in_uV=False)
        dmap = {int(u): dense[i] for i, u in enumerate(keep)}
        dense_by_t.append(dmap)
        if mask_ref is None:
            mask_ref = {int(u): np.flatnonzero(az.sparsity.mask[i]) for i, u in enumerate(keep)}
        else:
            for i, u in enumerate(keep):
                mask_ref.setdefault(int(u), np.flatnonzero(az.sparsity.mask[i]))

    # units present at ALL 5 points -> consecutive cosines
    allpres = np.all(present_by_t, axis=0)
    uids = np.asarray(asm.unit_ids)[allpres]
    print(f"units present at all 5 points: {len(uids)}/{asm.get_num_units()}", flush=True)
    min_cos = []
    for u in uids:
        ch = mask_ref[int(u)]
        coss = []
        for k in range(4):
            ta = dense_by_t[k][int(u)][:, ch]
            tb = dense_by_t[k + 1][int(u)][:, ch]
            coss.append(cosine_from_templates(ta, tb, max_shift_samples=10))
        min_cos.append(min(coss))
    min_cos = np.array(min_cos)
    print(f"per-unit MIN consecutive template cosine over 47h:", flush=True)
    print(f"  median={np.median(min_cos):.3f} p10={np.percentile(min_cos,10):.3f} "
          f"min={min_cos.min():.3f}", flush=True)
    for thr in (0.9, 0.8, 0.7, 0.5):
        print(f"  units with min-cos >= {thr}: {(min_cos>=thr).sum()}/{len(min_cos)} ({(min_cos>=thr).mean():.2f})", flush=True)
    print(f"\nstable (min-cos>=0.8 = smooth drift, no swap): {(min_cos>=0.8).mean():.0%}; "
          f"suspect (<0.7): {(min_cos<0.7).sum()} units", flush=True)
    np.savez(OUT / f"identity_check{tag}.npz", uids=uids, min_cos=min_cos, centers=centers)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

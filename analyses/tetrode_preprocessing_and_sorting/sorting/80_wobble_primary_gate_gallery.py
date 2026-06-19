"""Phase A': gate gallery -- what does each gate ADMIT to a fixed template? (per-gate rows + differential)

For a few well-isolated units across 2 windows, ONE figure per unit, 4-channel snippet (solid) over the
assigned template (gray dashed), with rows = the candidate gates so their qualitative selectivity is visible
(not just the bake-off's summary numbers, script 79):
  - circus [0.8,inf]                 -- the current production matcher's actual admits
  - wobble adaptive-||t||^2 @ 0.55x  -- the circus-matched absolute-threshold point
  - wobble @ intrinsic knee 0.68x    -- wobble's own absolute-threshold optimum (script 73)
  - wobble + cosine r>=r*            -- the chosen scale-invariant shape gate (r* from the finer sweep)
plus the DIFFERENTIAL admits, from ONE permissive 0.45x wobble run split by a=conv/||t||^2 vs r=cos(snip,t):
  - cosine-only   (a<0.8, r>=r*)  -- shape gate KEEPS, amplitude gate DROPS (low-amp template-shaped recoveries)
  - amplitude-only (a>=0.8, r<r*) -- amplitude gate KEEPS, shape gate DROPS (large wrong-shape mis-fits)
Per-gate rows are labelled with snippet MAD; the permissive-derived rows (cosine + differentials) also show
a and r. Reuses per_spike_fit + the 72 snippet/overlay style.

    cd gfys_workspace
    uv run --project /Users/gfindlay@ad.wisc.edu/projects/ece/gfys_workspace --extra tetrodes python \
        ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/sorting/80_wobble_primary_gate_gallery.py \
        [--windows-h 26 40] [--units 3] [--r-gate 0.60] [--matched-factor 0.55] [--knee-factor 0.68]
"""
import argparse
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.core import get_noise_levels

from _mp_common import (_unit_groups_from_mask, build_templates_object, materialize_span,
                        per_spike_fit, run_matching, tsq_median, wobble_method_kwargs)
from tetrode_analyses.tracking import sort_chunk, to_int_numpy_sorting

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/"
                   "track_eval/mp_long_s2000_d170000")
WV = OUT / "wobble_vs_circus"
FS = 30000.0
START_S, DUR_S = 2000.0, 170000.0
WIN_S = 1800.0
N_JOBS = 16
DETECT_THRESH = 5.5
NBEFORE, NAFTER = 30, 60
CIRCUS_KW = {"amplitudes": [0.8, np.inf]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-h", type=float, nargs="+", default=[26.0, 40.0])
    ap.add_argument("--units", type=int, default=3, help="units per window (top by circus admit count)")
    ap.add_argument("--r-gate", type=float, default=0.60, help="cosine shape gate r* (the finer-sweep knee)")
    ap.add_argument("--a-gate", type=float, default=0.80, help="amplitude gate (circus's a>=0.8) for the differential")
    ap.add_argument("--matched-factor", type=float, default=0.55, help="adaptive-||t||^2 circus-matched point")
    ap.add_argument("--knee-factor", type=float, default=0.68, help="adaptive-||t||^2 wobble intrinsic knee")
    ap.add_argument("--admit-factor", type=float, default=0.45, help="permissive admit for the cosine/diff rows")
    ap.add_argument("--n-ex", type=int, default=6, help="example panels per row")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    WV.mkdir(parents=True, exist_ok=True)
    rec = materialize_span(OUT, START_S, DUR_S)
    rec_groups = np.asarray(rec.get_property("group"))
    chan_ids = np.asarray(rec.channel_ids)

    for h in args.windows_h:
        a0 = int(h * 3600 * FS)
        b0 = min(a0 + int(WIN_S * FS), rec.get_num_frames())
        win = rec.frame_slice(a0, b0)
        win.reset_times()
        noise = get_noise_levels(win, return_in_uV=False)
        sdir = WV / "gallery" / f"w{int(h)}h_ref"
        shutil.rmtree(sdir, ignore_errors=True)
        ms5 = to_int_numpy_sorting(sort_chunk(win, sdir))
        bank, _ = build_templates_object(ms5, win, with_snr=False, n_jobs=N_JOBS, seed=0)
        ug = _unit_groups_from_mask(bank.sparsity.mask, rec_groups)
        dense = np.asarray(bank.get_dense_templates(), dtype=np.float32)
        nbefore_t = bank.nbefore
        med = tsq_median(bank)
        ids = np.asarray(bank.unit_ids)

        # the four gate runs + per-spike (a, r) on the permissive run
        def wb(f):
            return run_matching(win, bank, method="wobble", n_jobs=N_JOBS,
                                method_kwargs=wobble_method_kwargs(bank, threshold=f * med))[1]
        cs = run_matching(win, bank, method="circus-omp", method_kwargs=CIRCUS_KW, n_jobs=N_JOBS)[1]
        ad = wb(args.matched_factor)
        kn = wb(args.knee_factor)
        perm = wb(args.admit_factor)
        a_all, r_all = per_spike_fit(perm, bank, win)
        ps, pci = perm["sample_index"].astype(np.int64), perm["cluster_index"].astype(np.int64)
        co = (a_all < args.a_gate) & (r_all >= args.r_gate)   # cosine-only
        ao = (a_all >= args.a_gate) & (r_all < args.r_gate)   # amplitude-only
        cg = r_all >= args.r_gate                              # cosine gate admits
        # each ROW: (label, color, samples, cluster_idx, a-or-None, r-or-None)
        rows = [
            ("circus\n[0.8,inf]", "#d8743b", cs["sample_index"].astype(np.int64),
             cs["cluster_index"].astype(np.int64), None, None),
            (f"adaptive\n{args.matched_factor:.2f}x", "#8e44ad", ad["sample_index"].astype(np.int64),
             ad["cluster_index"].astype(np.int64), None, None),
            (f"knee\n{args.knee_factor:.2f}x", "#2e8b57", kn["sample_index"].astype(np.int64),
             kn["cluster_index"].astype(np.int64), None, None),
            (f"cosine\nr>={args.r_gate:.2f}", "#3b7dd8", ps[cg], pci[cg], a_all[cg], r_all[cg]),
            ("cosine-only\n(shape keeps,\namp drops)", "#3b7dd8", ps[co], pci[co], a_all[co], r_all[co]),
            ("amplitude-only\n(amp keeps,\nshape drops)", "#d8743b", ps[ao], pci[ao], a_all[ao], r_all[ao]),
        ]
        print(f"\n=== @ {h:.0f}h: {ids.size} units; admits circus {cs.size:,} / adaptive {ad.size:,} / "
              f"knee {kn.size:,} / permissive {perm.size:,} (cosine r>={args.r_gate} keeps {int(cg.sum()):,})",
              flush=True)

        # pick units: top by circus admit count (ensures the strictest row is populated)
        circ_counts = np.bincount(cs["cluster_index"].astype(np.int64), minlength=ids.size)
        picked = [u for u in np.argsort(circ_counts)[::-1][:args.units] if circ_counts[u] > 0]

        for u_idx in picked:
            g = int(ug[u_idx])
            chans = np.flatnonzero(rec_groups == g)
            templ = dense[u_idx][:, chans]

            def draw(ax, sample, color, a, r):
                tr = np.asarray(win.get_traces(start_frame=sample - NBEFORE, end_frame=sample + NAFTER,
                                               channel_ids=list(chan_ids[chans])), dtype=np.float32)
                mad = float(np.max(np.abs(tr) / noise[chans][None, :]))
                ymax = max(np.abs(tr).max(), np.abs(templ).max(), 1.0)
                off = 1.3 * ymax
                xt = np.arange(templ.shape[0]) + (NBEFORE - nbefore_t)
                for c in range(len(chans)):
                    ax.plot(tr[:, c] - c * off, color=color, lw=0.9)
                    ax.plot(xt, templ[:, c] - c * off, color="0.45", lw=0.8, ls="--")
                ax.axvline(NBEFORE, color="0.8", lw=0.6, zorder=0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-3.4 * off, 1.4 * off)
                ttl = f"{mad:.1f}MAD" + (f" a{a:.2f} r{r:.2f}" if a is not None else "")
                ax.set_title(ttl, fontsize=6.5, color=("#c0392b" if mad < DETECT_THRESH else "0.1"))

            fig, axes = plt.subplots(len(rows), args.n_ex, figsize=(2.0 * args.n_ex, 1.9 * len(rows)),
                                     squeeze=False)
            for ri, (label, color, s, ci, av, rv) in enumerate(rows):
                sel = np.flatnonzero(ci == u_idx)
                pick = rng.choice(sel, size=min(args.n_ex, sel.size), replace=False) if sel.size else np.empty(0, int)
                # map picked positions back to per-row a/r (av/rv are aligned to s/ci of this row)
                for c in range(args.n_ex):
                    ax = axes[ri][c]
                    if c < pick.size:
                        j = pick[c]
                        draw(ax, int(s[j]), color, (float(av[j]) if av is not None else None),
                             (float(rv[j]) if rv is not None else None))
                    else:
                        ax.axis("off")
                axes[ri][0].set_ylabel(f"{label}\n(n={sel.size})", fontsize=8, color=color,
                                       rotation=0, ha="right", va="center")
            fig.suptitle(
                f"Gate gallery u{ids[u_idx]} (tet {g}) @ {h:.0f}h -- snippet (solid) vs template (gray dashed)\n"
                f"rows 1-4 = what each GATE admits; rows 5-6 = DIFFERENTIAL (cosine-only vs amplitude-only)",
                fontsize=10)
            fig.tight_layout(rect=(0.06, 0, 1, 0.94))
            p = WV / f"gate_gallery_u{ids[u_idx]}_w{int(h)}h.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"  wrote {p.name} (circus n={int(circ_counts[u_idx])})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

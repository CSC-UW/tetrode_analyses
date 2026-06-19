"""Plots for the 48h carry-forward tracks (examine the results visually).

Produces (figures/ under the run dir):
  - continuity_trend.png : per-window present-fraction, fixed vs reestimate, over 48h.
  - track_span_hist.png  : per-unit tracked duration (windows present -> hours).
  - rate_heatmap.png     : firing rate (units x time), sorted by span -- each unit's activity over 48h.
  - template_evolution.png : peak-channel template at ~8 time points (colored by time) for a sample of
    STABLE and SUSPECT units (from identity_check.npz) -- shows drift vs possible swaps.

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/57_plot_carry_forward.py
"""
# ruff: noqa: E702  (compact one-line matplotlib plotting style, intentional in this figure script)
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si
from spikeinterface.core.template_tools import get_dense_templates_array

from _mp_common import build_templates_object

OUT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/track_eval/mp_long_s2000_d170000")
FS = 30000.0
WIN_S = 1800.0
N_TPTS = 8
TAG = ""               # set in main() from --tag (selects the assembled_reestimate<tag> variant)
FIG = OUT / "figures"  # recomputed in main() as figures<tag>
ASM_NAME = None        # assembled-sorting folder override (set in main; e.g. assembled_reseed_rs)
NPZ_NAME = None        # counts npz override (set in main; e.g. reseed_rs.npz, reseed format auto-detected)


def npz_plots():
    npz = np.load(OUT / (NPZ_NAME or f"long_drift{TAG}.npz"))
    if "counts" in npz.files:   # re-seeding output (reseed_rs.npz): counts is (n_units, n_windows)
        cnt = npz["counts"].T; fx = None
        pwr = (cnt >= 20).mean(1)
    else:
        fx = np.load(OUT / f"long_drift_fixed{TAG}.npz") if (OUT / f"long_drift_fixed{TAG}.npz").exists() else None
        pwr = npz["perwin_reest"]; cnt = npz["counts_reest"]
    nw = len(pwr); t_h = (np.arange(nw) + 0.5) * WIN_S / 3600.0

    # 1. continuity trend
    plt.figure(figsize=(8, 4))
    plt.plot(t_h, pwr, "-o", ms=3, label="reestimate")
    if fx is not None:
        plt.plot(t_h, fx["perwin_fixed"], "-s", ms=3, label="fixed", alpha=0.7)
    plt.axvspan(24, 29, color="orange", alpha=0.15, label="sleep deprivation (high movement)")
    plt.xlabel("time (h)"); plt.ylabel("fraction of confident units present (>=20 spk/30min)")
    plt.title("Carry-forward continuity over 48 h"); plt.ylim(0, 1.02); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "continuity_trend.png", dpi=120); plt.close()

    # 2. track-span histogram (hours each unit is present)
    span_h = (cnt >= 20).sum(0) * WIN_S / 3600.0
    plt.figure(figsize=(7, 4))
    plt.hist(span_h, bins=np.linspace(0, nw * WIN_S / 3600.0, 25), edgecolor="k")
    plt.axvline(np.median(span_h), color="r", ls="--", label=f"median {np.median(span_h):.0f} h")
    plt.xlabel("tracked duration (h)"); plt.ylabel("# units"); plt.legend()
    plt.title(f"Per-unit track span ({cnt.shape[1]} confident units)"); plt.tight_layout()
    plt.savefig(FIG / "track_span_hist.png", dpi=120); plt.close()

    # 3. firing-rate-over-time heatmap (units sorted by span)
    rate = cnt / WIN_S  # Hz
    order = np.argsort(-(cnt >= 20).sum(0))
    plt.figure(figsize=(10, 6))
    plt.imshow(rate[:, order].T, aspect="auto", origin="lower",
               extent=[0, nw * WIN_S / 3600.0, 0, cnt.shape[1]],
               cmap="magma", vmax=np.percentile(rate[rate > 0], 98))
    plt.colorbar(label="firing rate (Hz)"); plt.xlabel("time (h)"); plt.ylabel("unit (sorted by span)")
    plt.title("Tracked-unit activity over 48 h"); plt.tight_layout()
    plt.savefig(FIG / "rate_heatmap.png", dpi=120); plt.close()
    print(f"wrote continuity_trend / track_span_hist / rate_heatmap (median span {np.median(span_h):.0f} h)", flush=True)


def template_evolution():
    rec = si.load(OUT / "binary")
    asm = si.load(OUT / (ASM_NAME or f"assembled_reestimate{TAG}"))
    ic = np.load(OUT / f"identity_check{TAG}.npz")
    uids, mincos = ic["uids"], ic["min_cos"]
    order = np.argsort(mincos)
    suspect = uids[order[:4]]          # lowest min-cos = suspect drift/swap
    stable = uids[order[-4:]]          # highest = most stable
    sample = list(stable) + list(suspect)
    labels = [f"stable u{u}\n(mincos {mincos[order[-4:][i]]:.2f})" for i, u in enumerate(stable)] + \
             [f"SUSPECT u{u}\n(mincos {mincos[order[:4][i]]:.2f})" for i, u in enumerate(suspect)]

    total = rec.get_num_frames(); half = int(WIN_S * FS / 2)
    centers = np.linspace(half, total - half, N_TPTS).astype(int)
    # template per (timepoint, unit): dense + peak channel from first available
    tmpl = {int(u): {} for u in sample}
    peakch = {}
    for ti, c in enumerate(centers):
        s = asm.frame_slice(c - half, c + half)
        present = [u for u in sample if len(s.get_unit_spike_train(u)) >= 20]
        if not present:
            continue
        r = rec.frame_slice(c - half, c + half); r.reset_times()
        _, az = build_templates_object(s.select_units(present), r, with_snr=False)
        dense = get_dense_templates_array(az, return_in_uV=False)
        mask = az.sparsity.mask
        for i, u in enumerate(present):
            ch = np.flatnonzero(mask[i])
            tmpl[int(u)][ti] = dense[i][:, ch]
            if int(u) not in peakch:
                peakch[int(u)] = ch[np.argmax(np.ptp(dense[i][:, ch], axis=0))]  # peak channel (abs index)
                peakch[(int(u), "local")] = int(np.argmax(np.ptp(dense[i][:, ch], axis=0)))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    cmap = plt.get_cmap("viridis")
    for ax, u, lab in zip(axes.ravel(), sample, labels):
        d = tmpl[int(u)]
        for ti in sorted(d):
            wf = d[ti][:, peakch[(int(u), "local")]]
            ax.plot(wf, color=cmap(ti / (N_TPTS - 1)), lw=1.2)
        ax.set_title(lab, fontsize=9); ax.axhline(0, color="gray", lw=0.5)
    fig.suptitle("Peak-channel template evolution over 48 h (purple=0h -> yellow=47h)")
    fig.supxlabel("samples (1 ms before / 2 ms after peak)"); fig.supylabel("amplitude (raw)")
    fig.tight_layout()
    fig.savefig(FIG / "template_evolution.png", dpi=120); plt.close()
    print(f"wrote template_evolution (sample of {len(sample)} units, {N_TPTS} time points)", flush=True)


def main():
    global TAG, FIG, ASM_NAME, NPZ_NAME
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="suffix selecting a variant run (e.g. _dedup09); "
                    "reads long_drift<tag>.npz / assembled_reestimate<tag> / identity_check<tag>.npz, "
                    "writes to figures<tag>/")
    ap.add_argument("--assembled", default=None, help="assembled-sorting folder override (e.g. assembled_reseed_rs)")
    ap.add_argument("--npz", default=None, help="counts npz override (e.g. reseed_rs.npz; reseed format auto-detected)")
    args = ap.parse_args()
    TAG = args.tag; ASM_NAME = args.assembled; NPZ_NAME = args.npz
    FIG = OUT / f"figures{TAG}"
    FIG.mkdir(parents=True, exist_ok=True)
    npz_plots()
    template_evolution()
    print(f"\nfigures in {FIG}\nDONE", flush=True)


if __name__ == "__main__":
    main()

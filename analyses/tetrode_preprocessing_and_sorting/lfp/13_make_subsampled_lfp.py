"""Produce a 125 Hz subsampled LFP (16 tetrode leads) from the 625 Hz LFP.

Keeps one lead channel per tetrode (every 4th channel) and resamples the 625 Hz
LFP to 125 Hz (÷5, anti-aliased via resample's scipy.signal.decimate), float32,
microvolts, with the session-relative time vector carried through. Saves a
before/after trace + PSD plot for visual confirmation of the anti-alias rolloff.

    cd gfys_workspace
    uv run python ../tetrode_analyses/analyses/tetrode_preprocessing_and_sorting/lfp/13_make_subsampled_lfp.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from tetrode_analyses.lfp import make_subsampled_lfp, open_lfps_dataarray  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
LFP = ROOT / "2026-05-27_09-07-52.lfp.zarr"
SUB = ROOT / "2026-05-27_09-07-52.lfp.125hz.zarr"
OUTDIR = pathlib.Path(__file__).resolve().parent
RESAMPLE_RATE = 125

if not SUB.exists():
    make_subsampled_lfp(LFP, SUB, resample_rate=RESAMPLE_RATE, n_jobs=16)

parent = open_lfps_dataarray(LFP)
sub = open_lfps_dataarray(SUB)
print("sub-LFP DataArray:", sub.dims, sub.shape, "| fs", sub.attrs["fs"],
      "| dtype", sub.dtype, "| coords", list(sub.coords))
print("tetrodes:", list(np.asarray(sub["tetrode"].values)) if "tetrode" in sub.coords else "n/a")

# verification: 625 Hz lead channel vs its ~104 Hz version, a 4 s window 1 h in.
parent_fs = float(parent.attrs["fs"])
sub_fs = float(sub.attrs["fs"])
lead_id = str(sub["channel"].values[0])
t0 = 3600.0  # seconds into the recording

p_t = parent["time"].values
s_t = sub["time"].values
pi0 = int(np.searchsorted(p_t, p_t[0] + t0))
pi1 = int(np.searchsorted(p_t, p_t[0] + t0 + 4))
si0 = int(np.searchsorted(s_t, s_t[0] + t0))
si1 = int(np.searchsorted(s_t, s_t[0] + t0 + 4))

# select the same lead channel in both stores by channel name
p_lead = parent.sel(channel=lead_id).values[pi0:pi1]
s_lead = sub.sel(channel=lead_id).values[si0:si1]

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
axs[0].plot(np.arange(p_lead.size) / parent_fs, p_lead, lw=0.6, alpha=0.6,
            label=f"625 Hz ({lead_id})")
axs[0].plot(np.arange(s_lead.size) / sub_fs, s_lead, lw=1.4, color="C3",
            label=f"{sub_fs:.1f} Hz")
axs[0].set(title=f"625 Hz (faint) vs {sub_fs:.1f} Hz sub-LFP, 4 s @ t=1h",
           xlabel="s", ylabel="µV")
axs[0].legend()

# PSD on a 120 s chunk: parent vs sub, with the new Nyquist marked.
p120 = parent.sel(channel=lead_id).values[pi0:int(np.searchsorted(p_t, p_t[0] + t0 + 120))]
s120 = sub.sel(channel=lead_id).values[si0:int(np.searchsorted(s_t, s_t[0] + t0 + 120))]
fp, pp = signal.welch(p120, parent_fs, nperseg=4096)
fs_, ps_ = signal.welch(s120, sub_fs, nperseg=2048)
new_nyq = sub_fs / 2
axs[1].semilogy(fp, pp, lw=0.6, alpha=0.7, label="625 Hz LFP")
axs[1].semilogy(fs_, ps_, lw=1.2, color="C3", label=f"{sub_fs:.1f} Hz sub-LFP")
axs[1].axvline(new_nyq, color="k", ls="--", lw=0.8, label=f"sub Nyquist {new_nyq:.1f} Hz")
axs[1].set(xlim=(0, 312.5), xlabel="Hz", ylabel="PSD", title="PSD (anti-alias rolloff)")
axs[1].legend()
axs[2].semilogy(fs_, ps_, color="C3")
axs[2].set(xlim=(0, new_nyq), xlabel="Hz", ylabel="PSD",
           title=f"sub-LFP band detail (0–{new_nyq:.1f} Hz)")
fig.tight_layout()
fig.savefig(OUTDIR / "lfp_subsample_check.png", dpi=110)
print("saved", OUTDIR / "lfp_subsample_check.png")
print("DONE")

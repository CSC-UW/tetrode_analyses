"""Produce 625 Hz LFPs from the full-session blosc store + verification plots.

Two-stage anti-aliased decimation (30 kHz -> 3750 -> 625 Hz), float32, microvolts,
session-relative time vector carried through. Saves before/after trace + PSD
plots for visual confirmation.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import spikeinterface as si  # noqa: E402
from scipy import signal  # noqa: E402

from tetrode_analyses.lfp import make_lfp, open_lfps_dataarray  # noqa: E402

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
SRC = ROOT / "2026-05-27_09-07-52.blosc-zstd.zarr"
LFP = ROOT / "2026-05-27_09-07-52.lfp.zarr"
OUTDIR = pathlib.Path(__file__).resolve().parent
FS = 30000

if not LFP.exists():
    make_lfp(SRC, LFP, n_jobs=16)

da = open_lfps_dataarray(LFP)
print("LFP DataArray:", da.dims, da.shape, "| fs", da.attrs["fs"], "| dtype", da.dtype,
      "| coords", list(da.coords))

# verification plot: raw 30 kHz vs 625 Hz LFP for 2 channels, a 2 s window 1 h in
rec = si.read_zarr(str(SRC))
t0 = 3600  # seconds into the recording
raw = rec.get_traces(start_frame=t0 * FS, end_frame=(t0 + 2) * FS,
                     channel_ids=rec.get_channel_ids()[:2], return_in_uV=True)
lfp_win = da.isel(channel=slice(0, 2)).sel(time=slice(da.time.values[0], None))
# nearest LFP samples for the same 2 s window (session-relative time)
lfp_t = da.time.values
i0 = np.searchsorted(lfp_t, lfp_t[0] + t0)
i1 = np.searchsorted(lfp_t, lfp_t[0] + t0 + 2)

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
tr = np.arange(raw.shape[0]) / FS
tl = np.arange(i1 - i0) / 625
for ci in range(2):
    axs[0].plot(tr, raw[:, ci] + ci * 400, lw=0.4, alpha=0.5)
    axs[0].plot(tl, da.values[i0:i1, ci] + ci * 400, lw=1.2, color=f"C{ci+1}")
axs[0].set(title="raw 30 kHz (faint) vs 625 Hz LFP (bold), 2 ch offset, 2 s @ t=1h", xlabel="s", ylabel="µV")
# full-session PSD on a 60 s chunk
raw60 = rec.get_traces(start_frame=t0 * FS, end_frame=(t0 + 60) * FS,
                       channel_ids=[rec.get_channel_ids()[0]], return_in_uV=True)[:, 0]
lfp60 = da.values[i0:np.searchsorted(lfp_t, lfp_t[0] + t0 + 60), 0]
fr, pr = signal.welch(raw60, FS, nperseg=16384)
fl, pl = signal.welch(lfp60, 625, nperseg=4096)
axs[1].semilogy(fr, pr, lw=0.5, alpha=0.7, label="raw 30 kHz")
axs[1].semilogy(fl, pl, lw=1.0, color="C3", label="LFP 625 Hz")
axs[1].axvline(312.5, color="k", ls="--", lw=0.8, label="LFP Nyquist")
axs[1].set(xlim=(0, 1000), xlabel="Hz", ylabel="PSD", title="PSD (anti-alias rolloff)"); axs[1].legend()
axs[2].semilogy(fl, pl, color="C3"); axs[2].set(xlim=(0, 312.5), xlabel="Hz", ylabel="PSD", title="LFP band detail (0–312.5 Hz)")
fig.tight_layout()
fig.savefig(OUTDIR / "lfp_full_session_check.png", dpi=110)
print("saved", OUTDIR / "lfp_full_session_check.png")
print("DONE")

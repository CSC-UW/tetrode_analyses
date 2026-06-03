"""Compute per-tetrode STFT band-power timeseries and save to Zarr.

4 s DPSS STFT on the 625 Hz LFP decimated to 312.5 Hz (q=2), PSDs averaged within
tetrode, summed within each band (delta, vlad/eta, theta, sigma, gamma). Output:
``<subject>/<experiment>/stft_bandpowers.zarr`` (one ``(tetrode, time)`` var/band).
"""

from tetrode_analyses import experiment as exp
from tetrode_analyses import power
from tetrode_analyses.lfp import open_lfps_dataarray

SUBJECT = "TTM-001"
EXPERIMENT = "TTM-NOD"

params = exp.load_experiment_params(exp.experiment_params_path(SUBJECT, EXPERIMENT))
out = exp.experiment_dir(SUBJECT, EXPERIMENT) / "stft_bandpowers.zarr"

lfp = open_lfps_dataarray(params.lfp_zarr)
print(f"LFP {lfp.shape} @ {lfp.fs} Hz, {lfp.sizes['channel']} ch")

ds = power.compute_stft_bandpowers(lfp)
print(
    f"STFT band powers: bands={list(ds.data_vars)} "
    f"delta={ds['delta'].dims}{ds['delta'].shape} @ {ds.attrs['fs']} Hz "
    f"(n_fft={ds.attrs['stft_n_fft']})"
)
power.save_stft_bandpowers(ds, out)
print(f"Wrote {out}")

"""Extract per-tetrode instantaneous delta power (SWA) and save to Zarr.

Bandpass (0.5-4 Hz) + Hilbert envelope on the 625 Hz LFP, decimated to 125 Hz,
averaged within each tetrode (no bipolar referencing). Output:
``<subject>/<experiment>/delta.idelta.zarr`` (dims ``(tetrode, time)``).
"""

from dask.diagnostics import ProgressBar

from tetrode_analyses import experiment as exp
from tetrode_analyses import power
from tetrode_analyses.lfp import open_lfps_dataarray

SUBJECT = "TTM-001"
EXPERIMENT = "TTM-NOD"
LOWCUT, HIGHCUT = power.BANDS["delta"]

params = exp.load_experiment_params(exp.experiment_params_path(SUBJECT, EXPERIMENT))
out = exp.experiment_dir(SUBJECT, EXPERIMENT) / "delta.idelta.zarr"

lfp = open_lfps_dataarray(params.lfp_zarr)
print(f"LFP {lfp.shape} @ {lfp.fs} Hz, {lfp.sizes['channel']} ch")

ipwr = power.extract_instantaneous_bandpower(lfp, LOWCUT, HIGHCUT)
print(f"Computing instantaneous delta power -> {ipwr.dims} {ipwr.shape} @ {ipwr.fs} Hz")
with ProgressBar():
    power.save_instantaneous_power(ipwr, out)
print(f"Wrote {out}")

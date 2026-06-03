"""Plot SWA (delta power) across the whole recording with light/dark overlay.

For each SWA estimate (instantaneous delta power and STFT delta band power),
renders (a) a mean-across-tetrodes timetrace and (b) a per-tetrode small-multiples
figure, with the light/dark cycle shaded underneath (yellow = lights on, gray =
lights off) and the day-2 deprivation window shaded across. Figures are written to
``<subject>/<experiment>/figures/`` and copied next to this script.

The STFT trace uses the findlay2025a fig_2a params (smoothing=14, q=10); the
instantaneous trace scales those to its higher sample rate for a comparable look.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tetrode_analyses import experiment as exp  # noqa: E402
from tetrode_analyses import plotting, power  # noqa: E402

SUBJECT = "TTM-001"
EXPERIMENT = "TTM-NOD"
STFT_SMOOTHING, STFT_Q = 14, 10  # findlay2025a fig_2a params
# Display-only y-axis clip (on top of replace_outliers) so any residual transients
# don't compress the SWA trace.
YMAX_PCT = 99.5

exp_dir = exp.experiment_dir(SUBJECT, EXPERIMENT)
fig_dir = exp_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
local_dir = pathlib.Path(__file__).parent

params = exp.load_experiment_params(exp.experiment_params_path(SUBJECT, EXPERIMENT))
inst = power.open_instantaneous_power(exp_dir / "delta.idelta.zarr").compute()
stft = power.open_stft_bandpowers(exp_dir / "stft_bandpowers.zarr")["delta"].compute()

# Replace artifact outliers with NaN per tetrode at plot time (the saved products
# stay raw). Smoothing then skips the NaNs, so transients no longer dominate.
inst = power.replace_outliers_per_tetrode(inst)
stft = power.replace_outliers_per_tetrode(stft)


def _dt(da):
    return float(np.median(np.diff(da["time"].values)))


# Scale the instantaneous smoothing/decimation to match the STFT trace's
# wall-clock smoothing window and plot density.
scale = _dt(stft) / _dt(inst)
inst_smoothing, inst_q = round(STFT_SMOOTHING * scale), round(STFT_Q * scale)
print(f"STFT dt={_dt(stft):.3f}s  inst dt={_dt(inst):.4f}s  "
      f"-> inst smoothing={inst_smoothing}, q={inst_q}")

PRODUCTS = [
    ("inst", "Instantaneous delta power", inst, inst_smoothing, inst_q),
    ("stft", "STFT delta band power", stft, STFT_SMOOTHING, STFT_Q),
]


def save(fig, name):
    for d in (fig_dir, local_dir):
        fig.savefig(d / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}")


for key, label, da, smoothing, q in PRODUCTS:
    title = f"{SUBJECT} {EXPERIMENT} — SWA ({label})"

    fig, ax = plt.subplots(figsize=(14, 3))
    plotting.plot_swa_overview(
        da.mean("tetrode"), params, ax,
        smoothing=smoothing, q=q, title=title, ymax_percentile=YMAX_PCT,
    )
    save(fig, f"swa_{key}_mean.png")

    fig = plotting.plot_swa_small_multiples(
        da, params, smoothing=smoothing, q=q, title=title, ymax_percentile=YMAX_PCT
    )
    save(fig, f"swa_{key}_per_tetrode.png")

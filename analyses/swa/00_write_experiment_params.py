"""Write the WNE-style experiment_params.json for subject TTM-001 / TTM-NOD.

Records the light/dark cycle, the day-2 sleep-deprivation (novel-objects) window,
and the session anchor (parsed from the session slice_table). The session began
2026-05-27 09:07:52 America/Chicago (lights came on ~3 min earlier, at 09:05).
"""

import pathlib

from tetrode_analyses import experiment as exp

SUBJECT = "TTM-001"
EXPERIMENT = "TTM-NOD"
TIMEZONE = "America/Chicago"  # Madison/WISC local time; matches the OE session name

SESSION = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
LFP_ZARR = SESSION / "2026-05-27_09-07-52.lfp.zarr"

# Lights on 09:05, off 21:05 each day (12:12 cycle); deprivation 09:05-14:05 day 2.
LIGHTS_ON = ["2026-05-27T09:05:00", "2026-05-28T09:05:00", "2026-05-29T09:05:00"]
LIGHTS_OFF = ["2026-05-27T21:05:00", "2026-05-28T21:05:00"]
NOVEL_OBJECTS_START = "2026-05-28T09:05:00"
NOVEL_OBJECTS_END = "2026-05-28T14:05:00"

t0_unix = exp.read_session_t0_unix(SESSION / "slice_table.csv")

params = exp.ExperimentParams(
    subject=SUBJECT,
    experiment=EXPERIMENT,
    timezone=TIMEZONE,
    openephys_session=str(SESSION),
    lfp_zarr=str(LFP_ZARR),
    t0_unix=t0_unix,
    lightsOn=LIGHTS_ON,
    lightsOff=LIGHTS_OFF,
    novel_objects_start=NOVEL_OBJECTS_START,
    novel_objects_end=NOVEL_OBJECTS_END,
)

out = exp.experiment_params_path(SUBJECT, EXPERIMENT)
exp.save_experiment_params(params, out)

print(f"Wrote {out}")
print(f"Session anchor: {params.t0_datetime.isoformat()} (t0_unix={t0_unix})")
intervals, labels = exp.get_light_dark_periods(params)
print("Light/dark periods (session seconds):")
for (a, b), lab in zip(intervals, labels):
    print(f"  {lab:>3}  {a:>10.0f} -> {b:>10.0f}")
print(f"Deprivation (session seconds): {exp.get_deprivation_period(params)}")

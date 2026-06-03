"""Subject/experiment metadata and session-time mapping for tetrode recordings.

This module gives tetrode sessions the same WNE-style metadata contract that
SpikeGLX subjects get from ``ecephys.wne`` / ``wisc_ecephys_tools``, but
self-contained (no ``SGLXSubject`` dependency). An ``experiment_params.json``
file records the light/dark cycle, sleep-deprivation window, and the session
anchor; helpers map wall-clock datetimes to the session-relative seconds used by
the LFP ``time`` coordinate, and derive light/dark intervals for plotting.

The session anchor (``t0_unix``) is the Open Ephys "Software Time" of the first
recorded sample (see :func:`read_session_t0_unix`). Session ``time=0`` (the LFP
``time`` coordinate origin, as produced by
:func:`tetrode_analyses.spikeinterface.get_recording`) corresponds to this
instant, and later segments keep real wall-clock offsets, so a single wall-clock
subtraction maps any datetime to session seconds across the whole recording.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools as it
import json
import pathlib
import re
from zoneinfo import ZoneInfo

import pandas as pd

#: Local NVME staging root where tetrode subject data currently lives.
DEFAULT_DATA_ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data")

#: Filename WNE uses for per-experiment-subject parameters; reused here.
EXPERIMENT_PARAMS_FILENAME = "experiment_params.json"


@dataclasses.dataclass
class ExperimentParams:
    """Light/dark cycle, deprivation window, and session anchor for one subject.

    Datetime fields are naive ISO-8601 strings interpreted in ``timezone`` (the
    same encoding WNE uses). ``t0_unix`` anchors session-relative time.
    """

    subject: str
    experiment: str
    timezone: str
    openephys_session: str
    lfp_zarr: str
    t0_unix: float
    lightsOn: list[str]
    lightsOff: list[str]
    novel_objects_start: str | None = None
    novel_objects_end: str | None = None
    badChannels: list[str] = dataclasses.field(default_factory=list)
    badTetrodes: list[str] = dataclasses.field(default_factory=list)

    @property
    def t0_datetime(self) -> dt.datetime:
        """Session anchor as a timezone-aware datetime."""
        return dt.datetime.fromtimestamp(self.t0_unix, tz=ZoneInfo(self.timezone))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Provenance convenience: a human-readable anchor alongside t0_unix.
        d["t0_datetime"] = self.t0_datetime.isoformat()
        return d


def save_experiment_params(
    params: ExperimentParams, path: str | pathlib.Path
) -> pathlib.Path:
    """Write ``params`` to ``path`` as pretty-printed JSON, creating parents."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.to_dict(), indent=2) + "\n")
    return path


def load_experiment_params(path: str | pathlib.Path) -> ExperimentParams:
    """Load an ``experiment_params.json`` into an :class:`ExperimentParams`."""
    raw = json.loads(pathlib.Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(ExperimentParams)}
    return ExperimentParams(**{k: v for k, v in raw.items() if k in fields})


def experiment_dir(
    subject: str, experiment: str, root: str | pathlib.Path = DEFAULT_DATA_ROOT
) -> pathlib.Path:
    """Return ``<root>/<subject>/<experiment>`` (WNE-style experiment-subject dir)."""
    return pathlib.Path(root) / subject / experiment


def experiment_params_path(
    subject: str, experiment: str, root: str | pathlib.Path = DEFAULT_DATA_ROOT
) -> pathlib.Path:
    """Return the ``experiment_params.json`` path for a subject/experiment."""
    return experiment_dir(subject, experiment, root) / EXPERIMENT_PARAMS_FILENAME


def read_session_t0_unix(source: str | pathlib.Path) -> float:
    """Parse the session anchor (Unix epoch seconds) for an Open Ephys session.

    ``source`` may be:

    - a ``slice_table.csv`` written by ``tetrode_analyses.spikeinterface`` (uses
      the first segment's ``software_time_s``), or
    - an Open Ephys recording directory containing ``sync_messages.txt`` (parses
      the "Software Time (milliseconds since ... 1970 UTC)" line, matching
      :func:`tetrode_analyses.spikeinterface._read_software_time_s`).
    """
    source = pathlib.Path(source)
    if source.is_file() and source.suffix == ".csv":
        return float(pd.read_csv(source)["software_time_s"].iloc[0])
    sync = source / "sync_messages.txt" if source.is_dir() else source
    m = re.search(r"Software Time[^:]*:\s*(\d+)", sync.read_text())
    if not m:
        raise ValueError(f"No 'Software Time' line found in {sync}")
    return int(m.group(1)) / 1000.0


def dt2t(params: ExperimentParams, when: str | dt.datetime) -> float:
    """Map a wall-clock datetime (local to ``params.timezone``) to session seconds."""
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when)
    if when.tzinfo is None:
        when = when.replace(tzinfo=ZoneInfo(params.timezone))
    return when.timestamp() - params.t0_unix


def t2dt(params: ExperimentParams, t: float) -> dt.datetime:
    """Map session seconds back to a timezone-aware wall-clock datetime."""
    return dt.datetime.fromtimestamp(params.t0_unix + t, tz=ZoneInfo(params.timezone))


def get_light_dark_periods(
    params: ExperimentParams,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Return alternating light/dark ``(start, end)`` intervals in session seconds.

    Mirrors ``wisc_ecephys_tools.rats.cnd_hgs.get_light_dark_periods``: each
    interval's label is the transition at its start (``"on"`` = lights-on/light
    period, ``"off"`` = lights-off/dark period). Times are in session-relative
    seconds, so intervals before recording start are negative (they clip to the
    axis when plotted). The trailing period after the last transition has no
    closing transition and is therefore not returned.
    """
    transitions = [(dt2t(params, x), "on") for x in params.lightsOn]
    transitions += [(dt2t(params, x), "off") for x in params.lightsOff]
    transitions.sort()
    intervals = [(a[0], b[0]) for a, b in it.pairwise(transitions)]
    labels = [a[1] for a, _ in it.pairwise(transitions)]
    return intervals, labels


def get_deprivation_period(
    params: ExperimentParams,
) -> tuple[float, float] | None:
    """Return the ``(start, end)`` sleep-deprivation window in session seconds.

    Derived from ``novel_objects_start``/``novel_objects_end``; ``None`` if unset.
    """
    if params.novel_objects_start is None or params.novel_objects_end is None:
        return None
    return dt2t(params, params.novel_objects_start), dt2t(
        params, params.novel_objects_end
    )

"""Add curation-convenience unit properties to the tracked analyzer (for sigui filtering).

Writes four per-unit properties into the analyzer's ``sorting/properties/`` group so they
travel WITH the analyzer (and the curation bundle) and show up as sortable columns in
spikeinterface-gui's unit list -- no provenance file or extra plumbing needed at launch:

  * ``tier``        (str)  -- strictest isolation tier the unit passes:
                             conservative > moderate > permissive > none
                             (gate = _track_eval.isolation_tier_mask, SPOT).
  * ``tier_level``  (int8) -- 3/2/1/0 for the above (numeric -> sorts best-first).
  * ``n_chunks``    (int)  -- track span = number of distinct member chunks (provenance).
  * ``track_hours`` (float)-- approx tracked duration = (n_chunks+1)*stride_s/3600.

Idempotent: overwrites the four datasets if already present; leaves all other
analyzer/sorting data untouched (purely additive metadata). Run AFTER
37_build_analyzer_tracked.py (and re-run after any rebuild).

    cd gfys_workspace
    uv run --extra tetrodes python ../tetrode_analyses/.../sorting/45_add_curation_unit_properties.py
"""
import json
import pathlib

import numpy as np
import spikeinterface as si

from _track_eval import TIERS, isolation_tier_mask

ROOT = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52")
TRACKED = ROOT / "sortings_seed42_pcafix" / "tracked_48h"
ANALYZER = TRACKED / "analyzer_clustered.zarr"
PROV = TRACKED / "provenance_clustered.json"
STRIDE_S = 1800.0  # chunk_s=3600, overlap=0.5
TIER_LEVEL = {"permissive": 1, "moderate": 2, "conservative": 3}


def compute_properties(analyzer, provenance_path):
    """Return {name: array} of the four curation properties in analyzer.sorting unit order."""
    uids = list(analyzer.sorting.unit_ids)
    qm = analyzer.get_extension("quality_metrics").get_data().loc[uids]  # align to unit order

    tier = np.array(["none"] * len(uids), dtype="<U12")
    tier_level = np.zeros(len(uids), dtype=np.int8)
    for t in TIERS:  # permissive -> moderate -> conservative; strictest wins (tiers nested)
        m = isolation_tier_mask(qm, t)
        tier[m] = t
        tier_level[m] = TIER_LEVEL[t]

    prov = json.loads(pathlib.Path(provenance_path).read_text())
    span = {int(g): len({mem[1] for mem in members}) for g, members in prov.items()}
    n_chunks = np.array([span.get(int(u), 0) for u in uids], dtype=np.int32)
    track_hours = ((n_chunks.astype(np.float64) + 1) * STRIDE_S / 3600.0).astype(np.float32)
    return {"tier": tier, "tier_level": tier_level, "n_chunks": n_chunks, "track_hours": track_hours}


def write_properties(analyzer, props):
    """Persist props via the SortingAnalyzer API.

    ``set_sorting_property(..., save=True)`` writes each array into the zarr's
    ``sorting/properties/`` group AND re-consolidates the store metadata -- the step a
    direct zarr write misses, which is why the analyzer would otherwise ignore the
    properties on reload (it reads from the consolidated ``.zmetadata``).
    """
    for name, arr in props.items():
        analyzer.set_sorting_property(name, np.asarray(arr), save=True)


def main():
    analyzer = si.load_sorting_analyzer(str(ANALYZER))
    props = compute_properties(analyzer, PROV)
    write_properties(analyzer, props)

    # verify: reload fresh and confirm the new properties + a tier headcount
    re = si.load_sorting_analyzer(str(ANALYZER))
    from spikeinterface.widgets.utils import make_units_table_from_analyzer
    cols = set(make_units_table_from_analyzer(re).columns)
    for name in props:
        assert re.sorting.get_property(name) is not None, f"{name} did not persist"
        assert name in cols, f"{name} not in units table"
    tier = re.sorting.get_property("tier")
    counts = {t: int((tier == t).sum()) for t in ("conservative", "moderate", "permissive", "none")}
    nc = re.sorting.get_property("n_chunks")
    print(f"wrote {list(props)} -> {ANALYZER}", flush=True)
    print(f"tier counts (nested: cons<=mod<=perm): {counts}", flush=True)
    print(f"n_chunks: median={np.median(nc):.0f} max={int(nc.max())} | "
          f"conservative span median={np.median(nc[tier == 'conservative']):.0f} chunks", flush=True)
    print("units table columns now include:",
          [c for c in ("tier", "tier_level", "n_chunks", "track_hours") if c in cols], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

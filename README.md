# tetrode_analyses

Loading helpers for Open Ephys tetrode acquisitions.

## Public API (`tetrode_analyses` / `ta`)

| Function | Purpose |
| --- | --- |
| `print_session_tree(acq_dir)` | Print record nodes, experiments, recordings, and continuous streams (with channel/sample counts) for an acquisition. |
| `open_tetrode_dataarray(acq_dir, ...)` | Load one continuous stream as a tetrode-shaped lazy `xarray.DataArray` (`2d_flat_index`, `2d_multi_index`, or `3d` layout). |
| `load_channel_map(source, *, block_index=None)` | Load the tetrode channel map (`oe_indices`, `enabled`, `tt2ch_ixs`) from a `settings.xml` path, an acquisition / record-node directory, **or a SpikeInterface `OpenEphysBinaryRecordingExtractor`**. |
| `build_tetrode_probegroup(n_tetrodes, ...)` | Build a `probeinterface.ProbeGroup` of generic tetrodes (identity-wired). |
| `attach_tetrode_probegroup(recording, ...)` | Return a SpikeInterface recording reordered into channel-map order and grouped by tetrode (`group` + `tetrode` properties, plus a `ProbeGroup` when `geometry=True`). |

The channel map / tetrode grouping is an **experiment-level** property (read from
`settings.xml`'s `Channel Map` processor), while channel *names* are a
**per-recording** property (each recording's `structure.oebin`).

### Give a SpikeInterface extractor the tetrode grouping

A raw `OpenEphysBinaryRecordingExtractor` is a flat list of channels with no probe
and no grouping. `attach_tetrode_probegroup` applies the same channel map
`open_tetrode_dataarray` uses, so the recording can be split / preprocessed / sorted
per tetrode (cf. SpikeInterface's *Work with tetrodes* and *Process a recording by
channel group* how-tos):

```python
import tetrode_analyses as ta
from spikeinterface.extractors.extractor_classes import (
    OpenEphysBinaryRecordingExtractor,
)

extractor = OpenEphysBinaryRecordingExtractor(ACQ_DIR, stream_id="0")
grouped = ta.attach_tetrode_probegroup(extractor)

grouped.get_channel_groups()      # [0, 0, 0, 0, 1, 1, 1, 1, ...]
by_tetrode = grouped.split_by("group")   # one recording per tetrode
by_tetrode[0].get_channel_ids()   # ['CH40', 'CH38', 'CH36', 'CH34'] == TT1
```

`attach_tetrode_probegroup` / `build_tetrode_probegroup` require `probeinterface`,
declared in the optional `si` extra (`pip install -e '.[si]'` / `uv sync --extra si`).
The SpikeInterface extractor is supplied by the caller, so `spikeinterface` itself is
not a dependency of this package.

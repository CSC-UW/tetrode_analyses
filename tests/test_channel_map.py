"""Tests for the tetrode channel map: parsing, settings resolution, and the
SpikeInterface probegroup attachment."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tetrode_analyses.io import (
    _parse_channel_map_xml,
    _settings_filename,
    attach_tetrode_probegroup,
    build_tetrode_probegroup,
    load_channel_map,
)

# A minimal settings.xml with an 8-channel Channel Map → 2 tetrodes. The CH
# ``index`` values are deliberately reordered (not 0..7) to mirror a real map.
SETTINGS_XML = """<?xml version="1.0" ?>
<SETTINGS>
  <SIGNALCHAIN>
    <PROCESSOR name="Sources/Acquisition Board"/>
    <PROCESSOR name="Channel Map">
      <CUSTOM_PARAMETERS>
        <STREAM name="acquisition_board">
          <CH index="7" enabled="1"/>
          <CH index="5" enabled="1"/>
          <CH index="3" enabled="1"/>
          <CH index="1" enabled="1"/>
          <CH index="6" enabled="1"/>
          <CH index="4" enabled="0"/>
          <CH index="2" enabled="1"/>
          <CH index="0" enabled="1"/>
        </STREAM>
      </CUSTOM_PARAMETERS>
    </PROCESSOR>
  </SIGNALCHAIN>
</SETTINGS>
"""

EXPECTED_TT2CH = {1: [7, 5, 3, 1], 2: [6, 4, 2, 0]}


@pytest.fixture
def settings_path(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "settings.xml"
    p.write_text(SETTINGS_XML)
    return p


def test_parse_channel_map_xml(settings_path: pathlib.Path) -> None:
    cm = _parse_channel_map_xml(settings_path)
    assert cm["oe_indices"] == [7, 5, 3, 1, 6, 4, 2, 0]
    assert cm["enabled"] == [1, 1, 1, 1, 1, 0, 1, 1]
    assert cm["tt2ch_ixs"] == EXPECTED_TT2CH


def test_load_channel_map_from_file(settings_path: pathlib.Path) -> None:
    assert load_channel_map(settings_path)["tt2ch_ixs"] == EXPECTED_TT2CH


def test_load_channel_map_from_record_node_dir(settings_path: pathlib.Path) -> None:
    # A directory containing settings.xml is treated as a record-node directory.
    cm = load_channel_map(settings_path.parent)
    assert cm["tt2ch_ixs"] == EXPECTED_TT2CH


@pytest.mark.parametrize(
    ("experiment_name", "expected"),
    [
        ("experiment1", "settings.xml"),
        ("experiment2", "settings_2.xml"),
        ("experiment11", "settings_11.xml"),
    ],
)
def test_settings_filename(experiment_name: str, expected: str) -> None:
    assert _settings_filename(experiment_name) == expected


def test_build_tetrode_probegroup_wiring() -> None:
    pytest.importorskip("probeinterface")
    pg = build_tetrode_probegroup(3)
    probes = pg.probes
    assert len(probes) == 3
    assert sum(p.get_contact_count() for p in probes) == 12
    # Identity device wiring: contact i -> channel i.
    np.testing.assert_array_equal(
        pg.get_global_device_channel_indices()["device_channel_indices"],
        np.arange(12),
    )


def _toy_recording(channel_ids, tmp_path: pathlib.Path):
    """An 8-channel toy recording, saved to a binary folder so it is dumpable.

    ``set_probegroup`` round-trips the recording through a dict; an in-memory
    ``NumpyRecording`` would lose its custom string channel ids, so we persist it.
    """
    si = pytest.importorskip("spikeinterface.core")
    traces = np.zeros((100, len(channel_ids)), dtype="int16")
    rec = si.NumpyRecording(
        traces_list=[traces], sampling_frequency=30000.0, channel_ids=channel_ids
    )
    return rec.save(folder=tmp_path / "rec")


def test_attach_tetrode_probegroup_reorders_and_groups(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("probeinterface")
    rec = _toy_recording([f"CH{i + 1}" for i in range(8)], tmp_path)  # CH1..CH8
    channel_map = {
        "oe_indices": [7, 5, 3, 1, 6, 4, 2, 0],
        "enabled": [1] * 8,
        "tt2ch_ixs": EXPECTED_TT2CH,
    }

    grouped = attach_tetrode_probegroup(rec, channel_map)

    # Channels reordered into channel-map order: TT1 = indices 7,5,3,1 = CH8,CH6,CH4,CH2.
    assert list(map(str, grouped.get_channel_ids())) == [
        "CH8", "CH6", "CH4", "CH2",  # TT1
        "CH7", "CH5", "CH3", "CH1",  # TT2
    ]
    np.testing.assert_array_equal(
        grouped.get_channel_groups(), [0, 0, 0, 0, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        grouped.get_property("tetrode"),
        ["TT1", "TT1", "TT1", "TT1", "TT2", "TT2", "TT2", "TT2"],
    )

    by_group = grouped.split_by("group")
    assert len(by_group) == 2
    assert list(map(str, by_group[0].get_channel_ids())) == ["CH8", "CH6", "CH4", "CH2"]


def test_attach_tetrode_probegroup_group_property_only(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("probeinterface")
    rec = _toy_recording([f"CH{i + 1}" for i in range(8)], tmp_path)
    channel_map = {"tt2ch_ixs": EXPECTED_TT2CH}

    grouped = attach_tetrode_probegroup(rec, channel_map, geometry=False)
    np.testing.assert_array_equal(
        grouped.get_channel_groups(), [0, 0, 0, 0, 1, 1, 1, 1]
    )
    # No probe attached in group-only mode.
    assert not grouped.has_probe()


def test_attach_tetrode_probegroup_inconsistent_map_raises(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("probeinterface")
    rec = _toy_recording([f"CH{i + 1}" for i in range(8)], tmp_path)
    # Index 99 does not exist in an 8-channel recording.
    channel_map = {"tt2ch_ixs": {1: [0, 1, 2, 99]}}
    with pytest.raises(ValueError, match="inconsistent"):
        attach_tetrode_probegroup(rec, channel_map)


def test_attach_requires_channel_map_or_extractor(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("probeinterface")
    # Toy recording has no neo_reader.
    rec = _toy_recording([f"CH{i + 1}" for i in range(8)], tmp_path)
    with pytest.raises(ValueError, match="neo_reader"):
        attach_tetrode_probegroup(rec)


# --- End-to-end against the mounted acquisition (requires NFS) -----------------

ACQ_DIR = pathlib.Path(
    "/Volumes/neuropixel_archive/tetrode_data/2026-05-19_18-09-51"
)


@pytest.mark.requires_nfs
def test_extractor_grouping_matches_dataarray() -> None:
    pytest.importorskip("probeinterface")
    if not ACQ_DIR.exists():
        pytest.skip(f"acquisition not mounted: {ACQ_DIR}")
    from spikeinterface.extractors.extractor_classes import (
        OpenEphysBinaryRecordingExtractor,
    )

    import tetrode_analyses as ta

    extractor = OpenEphysBinaryRecordingExtractor(str(ACQ_DIR), stream_id="0")
    grouped = ta.attach_tetrode_probegroup(extractor)

    groups = grouped.get_channel_groups()
    assert len(set(groups)) == 16
    np.testing.assert_array_equal(groups, np.repeat(np.arange(16), 4))

    # TT1 channel order matches the ta DataArray's first tetrode.
    da = ta.open_tetrode_dataarray(ACQ_DIR, layout="2d_flat_index")
    tt1_names = list(da.channel.values[:4])
    by_group = grouped.split_by("group")
    assert list(map(str, by_group[0].get_channel_ids())) == tt1_names

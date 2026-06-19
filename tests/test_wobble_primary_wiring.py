"""Unit tests for the wobble-primary gating wiring in ``_mp_common``.

Covers the two pieces that make ``method="wobble"`` a production-selectable matcher alongside
``circus-omp`` (see MATCHING_PURSUIT_FINDINGS.md "Wobble as primary"):

  * ``_window_method_kwargs`` -- per-window matcher-kwargs routing: circus-omp kwargs pass through
    unchanged (scale-invariant gate), wobble derives a per-window threshold = factor * tsq_median.
  * ``per_spike_cosine`` -- the SCALE-INVARIANT cosine acceptance gate (``shape_gate_r``): a spike
    whose snippet equals its template scores r=1; a HALF-AMPLITUDE copy ALSO scores r=1 (the
    scale-invariance the whole approach rests on); an orthogonal snippet scores 0; an out-of-bounds
    snippet is NaN (so ``spikes[r >= r*]`` drops it).

These modules live in the analyses script dir, not the installed package, so it is added to sys.path.
"""
import pathlib
import sys

import numpy as np
import pytest

_SORTING_DIR = (pathlib.Path(__file__).resolve().parents[1]
                / "analyses" / "tetrode_preprocessing_and_sorting" / "sorting")
sys.path.insert(0, str(_SORTING_DIR))

import spikeinterface as si  # noqa: E402  (after sys.path setup)

from _mp_common import (  # noqa: E402
    _window_method_kwargs,
    per_spike_cosine,
    templates_from_dense,
    tsq_median,
)

FS = 30000.0
N_SAMP, NBEFORE, N_CHAN = 40, 15, 4


def _bank():
    """One-unit, single-tetrode (4 active channels) group-sparse RAW-unit Templates bank."""
    t = np.arange(N_SAMP) - NBEFORE
    w = (-100.0 * np.exp(-(t**2) / 8.0)).astype(np.float32)
    dense = np.zeros((1, N_SAMP, N_CHAN), dtype=np.float32)
    dense[0, :, 0] = w
    dense[0, :, 1] = 0.3 * w  # a little cross-channel spread, like a real tetrode footprint
    mask = np.ones((1, N_CHAN), dtype=bool)
    bank = templates_from_dense(dense, mask, NBEFORE, unit_ids=np.array([0]),
                                channel_ids=np.arange(N_CHAN))
    return bank, dense[0]


def _recording_with(template, placements):
    """NumpyRecording (4 ch, one tetrode group) with `template`-derived snippets placed at samples.

    placements: list of (peak_sample, snippet (N_SAMP, N_CHAN)). The snippet is written so that for a
    spike at peak_sample, per_spike_cosine's window tr[peak-NBEFORE : peak-NBEFORE+N_SAMP] == snippet.
    """
    total = max(p for p, _ in placements) + N_SAMP + 50
    traces = np.zeros((total, N_CHAN), dtype=np.float32)
    for peak, snip in placements:
        off = peak - NBEFORE
        traces[off:off + N_SAMP, :] = snip
    rec = si.NumpyRecording([traces], sampling_frequency=FS)
    rec.set_property("group", np.zeros(N_CHAN, dtype=int))  # all 4 channels = tetrode 0
    return rec


def _spikes(samples):
    arr = np.zeros(len(samples), dtype=[("sample_index", "int64"), ("cluster_index", "int64")])
    arr["sample_index"] = samples
    arr["cluster_index"] = 0
    return arr


# ---- _window_method_kwargs routing ------------------------------------------------------------

def test_circus_kwargs_pass_through_unchanged():
    bank, _ = _bank()
    circus = {"amplitudes": [0.8, float("inf")]}
    assert _window_method_kwargs("circus-omp", bank, method_kwargs=circus, wobble_factor=None) is circus


def test_wobble_threshold_is_factor_times_tsq_median():
    bank, _ = _bank()
    mk = _window_method_kwargs("wobble", bank, method_kwargs=None, wobble_factor=0.5)
    assert mk["parameters"]["threshold"] == pytest.approx(0.5 * tsq_median(bank))
    assert mk["parameters"]["approx_rank"] == 4  # clamped for a 4-channel tetrode template


def test_wobble_without_factor_raises():
    bank, _ = _bank()
    with pytest.raises(ValueError, match="wobble_factor"):
        _window_method_kwargs("wobble", bank, method_kwargs=None, wobble_factor=None)


def test_wobble_explicit_kwargs_pass_through():
    bank, _ = _bank()
    explicit = {"parameters": {"threshold": 123.0}}
    assert _window_method_kwargs("wobble", bank, method_kwargs=explicit, wobble_factor=None) is explicit


# ---- per_spike_cosine: the scale-invariant shape gate -----------------------------------------

def test_per_spike_cosine_gate():
    bank, templ = _bank()
    half = (0.5 * templ).astype(np.float32)                 # scale-invariance probe
    ortho = np.zeros_like(templ)
    ortho[:, 2] = templ[:, 0]                                # energy on a channel the template ignores
    pa, pb, pc = 500, 900, 1300
    rec = _recording_with(templ, [(pa, templ), (pb, half), (pc, ortho)])
    p_oob = rec.get_num_frames() + 5                         # snippet window runs off the end -> NaN
    spikes = _spikes([pa, pb, pc, p_oob])
    r = per_spike_cosine(spikes, bank, rec)

    assert r[0] == pytest.approx(1.0)          # exact match
    assert r[1] == pytest.approx(1.0)          # HALF amplitude -> still 1.0 (scale-invariant)
    assert r[2] == pytest.approx(0.0, abs=1e-6)  # orthogonal footprint
    assert np.isnan(r[3])                      # out-of-bounds snippet

    kept = spikes[r >= 0.5]                     # the run_matching shape_gate_r filter (NaN fails)
    assert set(kept["sample_index"].tolist()) == {pa, pb}

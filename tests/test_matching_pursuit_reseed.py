"""Unit tests for the periodic-re-seeding 'is this cluster new?' decision.

Covers ``tetrode_analyses.tracking.cluster_is_new``, the geometry-free template-competition rule that
drives ``_mp_common.windowed_carry_forward_reseed``: a re-sorted cluster becomes a NEW tracked unit iff
its 4-channel template matches no existing same-tetrode bank unit (max-shift cosine < add_cos). See
MATCHING_PURSUIT_FINDINGS.md "periodic re-seeding".
"""
import numpy as np
import pytest

from tetrode_analyses.tracking import cluster_is_new


def _wf(peak_ch, amp=-100.0, T=90, n_ch=4, shift=0):
    """Synthetic biphasic spike on one channel of a (T, 4) tetrode template, optionally time-shifted."""
    t = np.arange(T) - 30 - shift
    base = amp * np.exp(-(t**2) / 8.0) - 0.4 * amp * np.exp(-((t - 6) ** 2) / 20.0)
    out = np.zeros((T, n_ch), dtype=np.float32)
    out[:, peak_ch] = base
    out[:, (peak_ch + 1) % n_ch] = 0.3 * base  # a little cross-channel spread
    return out


def test_empty_bank_is_new():
    assert cluster_is_new(_wf(0), []) is True


def test_identical_template_not_new():
    t = _wf(2)
    assert cluster_is_new(t, [_wf(0), t.copy(), _wf(3)], add_cos=0.8) is False


def test_orthogonal_template_is_new():
    # peak on a different channel -> near-orthogonal -> below threshold -> NEW (the u20 late-vs-early case)
    assert cluster_is_new(_wf(1), [_wf(0), _wf(3)], add_cos=0.8) is True


def test_shifted_copy_matches_not_new():
    # same waveform shifted a few samples must still match (shift-tolerant cosine) -> not new
    assert cluster_is_new(_wf(0, shift=4), [_wf(0)], add_cos=0.8, max_shift_samples=10) is False


def test_threshold_is_respected():
    base = _wf(0)
    noisy = base + np.random.default_rng(0).normal(0, 8.0, base.shape).astype(np.float32)
    # a strict threshold should reject the noisy near-match as "new"; a lenient one should not
    assert cluster_is_new(noisy, [base], add_cos=0.999) is True
    assert cluster_is_new(noisy, [base], add_cos=0.5) is False


@pytest.mark.parametrize("add_cos", [0.7, 0.8, 0.9])
def test_self_is_never_new(add_cos):
    for ch in range(4):
        assert cluster_is_new(_wf(ch), [_wf(ch)], add_cos=add_cos) is False

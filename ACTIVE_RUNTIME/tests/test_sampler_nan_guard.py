"""Greedy decode must be a FUNCTION of the logits, even when they are not finite.

ACTIVE_RUNTIME/docs/cuda_port_record.md, item 1. The greedy branch of the MLX sampler used to run
`int(np.argmax(logits))` on the raw logits while the sampled branch was guarded.
That is backwards: greedy is the mode callers rely on to be reproducible, and
`np.argmax` over an array containing NaN returns the index of the FIRST NaN
instead of raising — so a single NaN silently picks a garbage token, and which
token that is moves with wherever the NaN landed.

These tests pin the guard's behaviour, not just its presence.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.mlx_dkv_wrapper import _sanitize_logits          # noqa: E402


def test_raw_argmax_really_does_pick_the_nan():
    """The bug this guard exists for — kept so the test is not vacuous."""
    x = np.array([1.0, np.nan, 3.0, 2.0], dtype=np.float32)
    assert int(np.argmax(x)) == 1          # the NaN, not the 3.0


def test_greedy_picks_the_true_max_through_a_nan():
    x = np.array([1.0, np.nan, 3.0, 2.0], dtype=np.float32)
    assert int(np.argmax(_sanitize_logits(x))) == 2


@pytest.mark.parametrize("nan_at", [0, 1, 500, 4095])
def test_greedy_is_position_invariant_to_where_the_nan_landed(nan_at):
    """The nondeterminism was that the CHOSEN TOKEN moved with the NaN's index."""
    x = np.full(4096, -5.0, dtype=np.float32)
    x[1234] = 9.0                          # the real argmax
    x[nan_at] = np.nan
    assert int(np.argmax(_sanitize_logits(x))) == 1234


def test_infinities_are_clamped_not_dropped():
    y = np.array([np.inf, -np.inf, np.nan, 0.5], dtype=np.float32)
    out = _sanitize_logits(y)
    assert np.isfinite(out).all()
    assert out.tolist() == [100.0, -100.0, -100.0, 0.5]
    assert int(np.argmax(out)) == 0        # +inf stays the largest


def test_all_nan_is_deterministic_rather_than_arbitrary():
    x = np.full(64, np.nan, dtype=np.float32)
    out = _sanitize_logits(x)
    assert np.isfinite(out).all()
    assert int(np.argmax(out)) == 0        # a defined answer, the same every run


def test_finite_input_is_returned_untouched():
    """The common path must not copy, and must not perturb any value."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _sanitize_logits(x) is x


def test_sampled_path_stays_drawable_after_sanitising():
    """np.random.choice raises on NaN probabilities; the guard must prevent that."""
    x = np.array([1.0, np.nan, 3.0, 2.0], dtype=np.float32)
    scaled = _sanitize_logits(x) / 0.7
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    int(np.random.choice(len(probs), p=probs))       # must not raise


def test_warning_fires_once_per_owner():
    class Owner:
        pass
    o = Owner()
    bad = np.array([np.nan, 1.0], dtype=np.float32)
    _sanitize_logits(bad, o)
    assert o._nonfinite_logits_warned is True
    _sanitize_logits(bad, o)                          # second call must not re-warn

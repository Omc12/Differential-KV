"""Unit tests for Rarity-Aware Residual Selection (CAD Upgrade).

Validates:
1. IDF formula: IDF(t) = ln(1 + N / count(t))
2. Exclusion guard: delimiters/punctuation are strictly filtered (1.0x boost).
3. Token boost multipliers:
   - Digits: 20.0x
   - TitleCase / Owner Names: 14.6x
   - Rare Words (IDF >= 2.0): 7.3x (1.0 + weight * IDF)
   - Delimiters / Punctuation: 1.0x (no boost)
4. Environment controls: DKV_RARITY_CAPTURE, DKV_RARITY_WEIGHT, DKV_RARITY_MIN_IDF.
"""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.compression.residual_capture import (
    _compute_idf,
    _apply_rarity_capture,
    compute_boost_multipliers,
)


def test_idf_formula():
    # IDF(t) = ln(max(N, 2) / (count(t) + 0.1)).
    # The +0.1 smoothing and the dropped 1+ landed in 46737636, which aligned
    # this path with the six MLX call sites in mlx_dkv_wrapper.py; the two
    # engines must not diverge on the rarity score or they select different
    # residual rows for the same block.
    counts = {101: 1, 102: 10, 103: 1000}
    N = 16000
    idf_rare = _compute_idf(101, counts, N)
    assert math.isclose(idf_rare, math.log(16000.0 / 1.1), rel_tol=1e-5)

    idf_mid = _compute_idf(102, counts, N)
    assert math.isclose(idf_mid, math.log(16000.0 / 10.1), rel_tol=1e-5)

    idf_freq = _compute_idf(103, counts, N)
    assert math.isclose(idf_freq, math.log(16000.0 / 1000.1), rel_tol=1e-5)

    # Monotone in rarity, which is the property the selection actually relies on.
    assert idf_rare > idf_mid > idf_freq


def test_exclusion_guard_delimiters():
    delims = ["{", "}", "[", "]", ":", "\n", ",", ".", "   ", ";"]
    tids = list(range(len(delims)))
    counts = {i: 1 for i in tids}  # even if counts are 1 (nominally rare)
    boost = [1.0] * len(delims)
    
    n_boosted = _apply_rarity_capture(
        boost, delims, tids, counts, total_tokens=16000,
        rarity_weight=1.5, min_idf=2.0, target_mult=7.3
    )
    assert n_boosted == 0
    assert all(b == 1.0 for b in boost)


def test_rarity_multiplier_on_rare_words():
    tok_strs = ["quantum", "fluctuation", "the", "and", "{", "}"]
    tids = [1, 2, 3, 4, 5, 6]
    # quantum and fluctuation appear once, 'the' appears 5000 times, and delims are punctuation
    counts = {1: 1, 2: 1, 3: 5000, 4: 5000, 5: 10000, 6: 10000}
    total_tokens = 16000
    boost = [1.0] * len(tok_strs)

    n_boosted = _apply_rarity_capture(
        boost, tok_strs, tids, counts, total_tokens,
        rarity_weight=1.5, min_idf=2.0, target_mult=7.3
    )
    assert n_boosted == 2
    # Rare words get >= 7.3x boost
    assert boost[0] >= 7.3
    assert boost[1] >= 7.3
    # Common words stay 1.0 (IDF < 2.0)
    assert boost[2] == 1.0
    assert boost[3] == 1.0
    # Delimiters stay 1.0
    assert boost[4] == 1.0
    assert boost[5] == 1.0


def test_multipliers_digits_and_owner():
    # Prompt:
    # Digits: 20.0x
    # TitleCase / Owner: 14.6x
    tok_strs = ["Dr.", "Kestrel", "found", "code", "942718", "in", "the", "lab", ".\n"]
    tids = list(range(len(tok_strs)))
    counts = {i: 100 for i in tids}
    total_tokens = 16000

    boost, n = compute_boost_multipliers(tok_strs, tids, counts, total_tokens)
    assert boost is not None and n > 0

    digit_idx = tok_strs.index("942718")
    assert boost[digit_idx] >= 20.0, f"Digit boost expected >= 20.0, got {boost[digit_idx]}"

    owner_idx = tok_strs.index("Kestrel")
    assert boost[owner_idx] >= 14.6, f"Owner boost expected >= 14.6, got {boost[owner_idx]}"


def test_rarity_env_controls(monkeypatch):
    tok_strs = ["exotic_entity", "normal_word", "the"]
    tids = [10, 20, 30]
    counts = {10: 1, 20: 100, 30: 5000}
    total_tokens = 16000

    # Test DKV_RARITY_CAPTURE=0 disables rarity boost
    monkeypatch.setenv("DKV_RARITY_CAPTURE", "0")
    b_out, _ = compute_boost_multipliers(tok_strs, tids, counts, total_tokens)
    assert b_out[0] == 1.0, "Rarity boost should be disabled when DKV_RARITY_CAPTURE=0"

    # Test DKV_RARITY_CAPTURE=1 enables rarity boost
    monkeypatch.setenv("DKV_RARITY_CAPTURE", "1")
    monkeypatch.setenv("DKV_RARITY_MIN_IDF", "2.0")
    monkeypatch.setenv("DKV_RARITY_WEIGHT", "2.0")
    b_on, _ = compute_boost_multipliers(tok_strs, tids, counts, total_tokens)
    assert b_on[0] >= 7.3, f"Rarity boost expected >= 7.3, got {b_on[0]}"

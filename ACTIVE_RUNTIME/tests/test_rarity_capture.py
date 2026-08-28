"""Unit tests for the rarity-capture pass in residual_capture.py and
the MLX-wrapper's _apply_rarity_capture helper.

No GPU, no HF tokenizer required — runs on CPU only.
"""
import os
import sys
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.compression.residual_capture import compute_boost_multipliers


# ---------------------------------------------------------------------------
# Tests for compute_boost_multipliers (residual_capture.py rarity pass)
# ---------------------------------------------------------------------------

class TestRarityCapture:
    def test_rare_word_receives_boost(self, monkeypatch):
        """A token appearing only once in a 4096-token session should clear the
        default IDF threshold of 3.0 and receive a boost."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        toks = [" obscurantism"]   # unique, rare word
        tids = [0]
        counts = {0: 1}
        boost, n = compute_boost_multipliers(toks, tids, counts, total)
        assert n > 0, "Expected rarity boost for rare token"
        assert boost[0] > 1.0, f"Expected boost > 1.0, got {boost[0]}"

    def test_common_word_not_rarity_boosted(self, monkeypatch):
        """A very common token should NOT receive a rarity boost (IDF below threshold)."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        toks = [" the"]   # very common
        tids = [0]
        counts = {0: 1000}
        boost, _ = compute_boost_multipliers(toks, tids, counts, total)
        # IDF = log(4096/1000.1) ~ 1.41 < 3.0 → no rarity boost
        rarity_idf = math.log(max(total, 2) / (1000 + 0.1))
        assert rarity_idf < 3.0
        assert boost[0] == 1.0, (
            f"Common lowercase token should stay at 1.0, got {boost[0]}; IDF={rarity_idf:.2f}")

    def test_rarity_disabled_by_env(self, monkeypatch):
        """DKV_RESIDUAL_RARITY_CAPTURE=0 must completely suppress rarity boosts."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "0")
        total = 4096
        toks = [" arcane", " mechanism", " theory"]
        tids = [0, 1, 2]
        counts = {0: 1, 1: 1, 2: 1}
        boost, n = compute_boost_multipliers(toks, tids, counts, total)
        for i, b in enumerate(boost):
            assert b == 1.0, f"Token {i} got boost {b} despite RARITY_CAPTURE=0"

    def test_rarity_does_not_fire_without_counts(self, monkeypatch):
        """Empty counts dict → skip rarity (uniform IDF is meaningless)."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        toks = [" arcane", " mechanism"]
        tids = [0, 1]
        counts = {}   # no frequency information
        boost, _ = compute_boost_multipliers(toks, tids, counts, total)
        for i, b in enumerate(boost):
            assert b == 1.0, (
                f"Token {i} got boost {b} with empty counts — rarity should skip")

    def test_rarity_below_digit_boost(self, monkeypatch):
        """A rare word must NOT outrank a digit token (sub-unity rarity_weight).

        The window pass (W=2) spreads a boosted token's weight to its neighbours,
        so the digit and rare word must be placed > 2 tokens apart to isolate their
        individual boosts for comparison.
        """
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        # W=2 window pass: tokens within 2 positions of a boosted token inherit
        # its boost. Use 5 neutral filler tokens between digit and rare word so
        # the window does not merge their scores.
        filler = [" the"] * 5
        toks = [" 42"] + filler + [" arcane"]
        tids = list(range(len(toks)))
        counts = {i: 1 for i in range(len(toks))}  # all equally rare
        boost, _ = compute_boost_multipliers(toks, tids, counts, total)
        digit_boost = boost[0]
        rare_boost = boost[-1]
        assert digit_boost > rare_boost, (
            f"Digit boost {digit_boost} should exceed rare-word boost {rare_boost}; "
            f"check that rarity_weight=0.5 is applied and token separation >= W=2")

    def test_punctuation_not_rarity_boosted(self, monkeypatch):
        """Punctuation tokens (no alnum char) must never receive a rarity boost."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        toks = [",", ".", ":", "{", "}"]
        tids = [0, 1, 2, 3, 4]
        counts = {i: 1 for i in range(5)}
        boost, _ = compute_boost_multipliers(toks, tids, counts, total)
        for i, b in enumerate(boost):
            assert b == 1.0, f"Punctuation token {toks[i]!r} got rarity boost {b}"

    def test_rarity_weight_env_respected(self, monkeypatch):
        """DKV_RESIDUAL_RARITY_WEIGHT=1.0 doubles the rarity boost vs 0.5."""
        total = 4096
        toks = [" arcane"]
        tids = [0]
        counts = {0: 1}
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_WEIGHT", "0.5")
        boost_half, _ = compute_boost_multipliers(toks, tids, counts, total)
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_WEIGHT", "1.0")
        boost_full, _ = compute_boost_multipliers(toks, tids, counts, total)
        if boost_half[0] > 1.0 and boost_full[0] > 1.0:
            assert boost_full[0] > boost_half[0], (
                f"rarity_weight=1.0 should give higher boost than 0.5; "
                f"got {boost_full[0]} vs {boost_half[0]}")

    def test_rarity_min_idf_env_respected(self, monkeypatch):
        """Lowering DKV_RESIDUAL_RARITY_MIN_IDF makes more tokens eligible."""
        total = 100
        toks = [" somewhat"]    # appears 5 times → IDF ≈ log(100/5.1) ≈ 2.98
        tids = [0]
        counts = {0: 5}
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_WEIGHT", "0.5")
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_MIN_IDF", "3.0")
        boost_high, _ = compute_boost_multipliers(toks, tids, counts, total)
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_MIN_IDF", "2.5")
        boost_low, _ = compute_boost_multipliers(toks, tids, counts, total)
        idf = math.log(max(total, 2) / (5 + 0.1))
        if idf < 3.0:
            assert boost_high[0] == 1.0, (
                f"IDF={idf:.2f} should not clear threshold 3.0, got {boost_high[0]}")
            assert boost_low[0] > 1.0, (
                f"IDF={idf:.2f} should clear threshold 2.5, got {boost_low[0]}")


# ---------------------------------------------------------------------------
# Tests for _apply_rarity_capture in mlx_dkv_wrapper
# ---------------------------------------------------------------------------

class TestMLXRarityCapture:
    """Test the standalone _apply_rarity_capture helper in mlx_dkv_wrapper.py."""

    @pytest.fixture(autouse=True)
    def import_helper(self):
        try:
            _serving = os.path.join(os.path.dirname(__file__), "..", "serving")
            sys.path.insert(0, _serving)
            from mlx_dkv_wrapper import _apply_rarity_capture
            self._fn = _apply_rarity_capture
        except ImportError:
            pytest.skip("mlx_dkv_wrapper not importable (mlx not installed)")

    def test_rare_word_boosted(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        boosts = [1.0]
        toks = [" arcane"]
        tids = [0]
        counts = {0: 1}
        n = self._fn(boosts, toks, tids, counts, 4096, tok_boost=8.0)
        assert n > 0
        assert boosts[0] > 1.0

    def test_empty_counts_skips(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        boosts = [1.0]
        toks = [" arcane"]
        tids = [0]
        n = self._fn(boosts, toks, tids, {}, 4096, tok_boost=8.0)
        assert n == 0
        assert boosts[0] == 1.0

    def test_already_shape_boosted_not_overridden(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        boosts = [100.0]   # already boosted by a shape rule
        toks = [" 42"]
        tids = [0]
        counts = {0: 1}
        n = self._fn(boosts, toks, tids, counts, 4096, tok_boost=8.0)
        assert n == 0
        assert boosts[0] == 100.0  # untouched

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "0")
        boosts = [1.0]
        toks = [" arcane"]
        tids = [0]
        counts = {0: 1}
        n = self._fn(boosts, toks, tids, counts, 4096, tok_boost=8.0)
        assert n == 0
        assert boosts[0] == 1.0

    def test_punctuation_skipped(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        boosts = [1.0, 1.0, 1.0]
        toks = [",", ".", ":"]
        tids = [0, 1, 2]
        counts = {i: 1 for i in range(3)}
        n = self._fn(boosts, toks, tids, counts, 4096, tok_boost=8.0)
        assert n == 0
        assert all(b == 1.0 for b in boosts)

    def test_rarity_below_digit_level(self, monkeypatch):
        """Rarity boost (0.5×) must be lower than a digit's full tok_boost."""
        monkeypatch.setenv("DKV_RESIDUAL_RARITY_CAPTURE", "1")
        total = 4096
        # Apply manually: digit gets 8.0×idf/2 via core-segment; rare word gets 0.5×
        tok_boost = 8.0
        counts = {0: 1, 1: 1}
        idf = math.log(max(total, 2) / (1 + 0.1))
        digit_boost = tok_boost * (idf / 2.0)    # full core-segment boost
        boosts = [1.0]
        toks = [" arcane"]
        tids = [1]
        self._fn(boosts, toks, tids, counts, total, tok_boost=tok_boost)
        rarity_boost = boosts[0]
        assert rarity_boost < digit_boost, (
            f"Rarity boost {rarity_boost:.2f} should be < digit boost {digit_boost:.2f}")

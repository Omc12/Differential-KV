"""
tests/test_anchor_logic.py

Unit tests for anchor placement strategies and AnchorManager.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor_logic.strategies import (
    PeriodicAnchorStrategy, AdaptiveAnchorStrategy, AnchorDecision
)
from anchor_logic.anchor_manager import AnchorManager


# ── PeriodicAnchorStrategy ────────────────────────────────────────────────────

class TestPeriodicAnchorStrategy:

    def test_first_anchor_after_interval(self):
        strat = PeriodicAnchorStrategy(interval=64)
        torch.manual_seed(0)
        kv  = torch.randn(2, 32, 128, dtype=torch.float16)
        anc = torch.randn(2, 32, 128, dtype=torch.float16)

        # Token 63: 63 tokens since anchor 0 → NOT yet
        d = strat.should_anchor(63, kv, anc, last_anchor_idx=0)
        assert not d.is_anchor

        # Token 64: exactly at interval → IS anchor
        d = strat.should_anchor(64, kv, anc, last_anchor_idx=0)
        assert d.is_anchor
        assert d.reason == "periodic"

    def test_anchors_at_multiples_of_interval(self):
        strat = PeriodicAnchorStrategy(interval=32)
        torch.manual_seed(1)
        kv  = torch.randn(2, 32, 128)
        anc = torch.zeros_like(kv)

        # Simulate: last anchor at 0, current at 31 → not anchor
        d = strat.should_anchor(31, kv, anc, 0)
        assert not d.is_anchor

        # Current at 32 → anchor
        d = strat.should_anchor(32, kv, anc, 0)
        assert d.is_anchor

    def test_invalid_interval_raises(self):
        with pytest.raises(AssertionError):
            PeriodicAnchorStrategy(interval=0)


# ── AdaptiveAnchorStrategy ────────────────────────────────────────────────────

class TestAdaptiveAnchorStrategy:

    def setup_method(self):
        self.strat = AdaptiveAnchorStrategy(
            max_interval=64,
            delta_norm_threshold=2.0,
            error_estimate_threshold=0.05,
            min_interval=8,
        )

    def test_no_anchor_within_min_interval(self):
        torch.manual_seed(0)
        kv  = torch.randn(2, 32, 128) * 100.0   # large delta
        anc = torch.zeros_like(kv)

        # 5 tokens since last anchor → should NOT anchor (within min_interval=8)
        d = self.strat.should_anchor(5, kv, anc, last_anchor_idx=0)
        assert not d.is_anchor

    def test_periodic_hard_limit(self):
        torch.manual_seed(0)
        kv  = torch.randn(2, 32, 128) * 0.0001   # tiny delta
        anc = kv.clone()

        # 64 tokens since last anchor → must anchor (periodic)
        d = self.strat.should_anchor(64, kv, anc, last_anchor_idx=0)
        assert d.is_anchor
        assert d.reason == "periodic"

    def test_magnitude_trigger(self):
        torch.manual_seed(0)
        anchor_kv = torch.zeros(2, 32, 128)
        big_kv    = torch.ones(2, 32, 128) * 10.0   # huge delta norm

        # 20 tokens since last anchor → magnitude trigger
        d = self.strat.should_anchor(20, big_kv, anchor_kv, last_anchor_idx=0)
        assert d.is_anchor
        assert d.reason == "adaptive_magnitude"

    def test_smooth_kv_avoids_early_anchors(self):
        """Nearly identical KV should NOT trigger early anchor."""
        torch.manual_seed(0)
        anchor_kv = torch.ones(2, 32, 128)
        smooth_kv = anchor_kv + torch.randn(2, 32, 128) * 0.001   # tiny noise

        d = self.strat.should_anchor(20, smooth_kv, anchor_kv, last_anchor_idx=0)
        assert not d.is_anchor


# ── AnchorManager ─────────────────────────────────────────────────────────────

class TestAnchorManager:

    def _make_kv_seq(self, seq_len=256, seed=0):
        torch.manual_seed(seed)
        return torch.randn(seq_len, 2, 8, 32, dtype=torch.float16)

    def test_token_zero_always_anchor(self):
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=64))
        kv = self._make_kv_seq(256)
        manager.compress(kv)
        assert 0 in manager.anchor_table

    def test_periodic_anchors_at_correct_positions(self):
        interval = 32
        seq_len  = 128
        manager  = AnchorManager(strategy=PeriodicAnchorStrategy(interval=interval))
        kv       = self._make_kv_seq(seq_len)
        manager.compress(kv)

        expected = {0, 32, 64, 96}   # token 0 + every 32
        assert expected.issubset(set(manager.anchor_table.keys()))

    def test_anchor_table_plus_delta_table_covers_all_tokens(self):
        seq_len = 100
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=16))
        kv = self._make_kv_seq(seq_len)
        manager.compress(kv)

        all_stored = set(manager.anchor_table.keys()) | set(manager.delta_table.keys())
        assert all_stored == set(range(seq_len)), \
            f"Missing tokens: {set(range(seq_len)) - all_stored}"

    def test_compression_ratio_greater_than_one_for_smooth_kv(self):
        """Smooth KV should always compress better than raw FP16."""
        import math
        # Slowly varying KV
        seq_len = 512
        t = torch.linspace(0, 2 * math.pi, seq_len)
        kv = torch.sin(t).view(seq_len, 1, 1, 1).expand(seq_len, 2, 8, 32).clone()
        kv = kv.to(torch.float16)

        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=64))
        stats   = manager.compress(kv)
        assert stats.compression_ratio > 1.0, \
            f"Expected compression > 1x, got {stats.compression_ratio:.3f}"

    def test_get_preceding_anchor_returns_correct(self):
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=32))
        kv = self._make_kv_seq(128)
        manager.compress(kv)

        # Token 50 should have anchor at 32
        anchor_idx, anchor_kv = manager.get_preceding_anchor(50)
        assert anchor_idx == 32

        # Token 0 should have anchor at 0
        anchor_idx, _ = manager.get_preceding_anchor(0)
        assert anchor_idx == 0

    def test_is_anchor_correct(self):
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=32))
        kv = self._make_kv_seq(128)
        manager.compress(kv)

        assert manager.is_anchor(0)
        assert manager.is_anchor(32)
        assert not manager.is_anchor(15)
        assert not manager.is_anchor(47)

    def test_compression_stats_sum_correct(self):
        seq_len = 200
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=50))
        kv = self._make_kv_seq(seq_len)
        stats = manager.compress(kv)

        assert stats.num_tokens == seq_len
        assert stats.num_anchors + stats.num_deltas == seq_len
        assert stats.original_fp16_bytes > 0
        assert stats.total_compressed_bytes > 0

    def test_reset_clears_state(self):
        manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=32))
        kv = self._make_kv_seq(128)
        manager.compress(kv)
        manager.reset()

        assert len(manager.anchor_table) == 0
        assert len(manager.delta_table)  == 0
        assert len(manager.index_list)   == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

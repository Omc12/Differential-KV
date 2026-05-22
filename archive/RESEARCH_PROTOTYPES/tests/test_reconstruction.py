"""
tests/test_reconstruction.py

Unit tests for KVReconstructor.

Tests:
  - Single token reconstruction correctness
  - Grouped range reconstruction correctness
  - Error measurement API
  - Reconstruction from anchor vs delta
  - Edge cases: first/last token, single-token range, full-sequence range
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from reconstruction.reconstructor import KVReconstructor


HEADS    = 8
HEAD_DIM = 32
SEQ_LEN  = 128
INTERVAL = 32


@pytest.fixture
def compressed_manager():
    """Returns (manager, original_kv) with periodic anchors every 32 tokens."""
    torch.manual_seed(99)
    kv = torch.randn(SEQ_LEN, 2, HEADS, HEAD_DIM, dtype=torch.float16)
    manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=INTERVAL))
    manager.compress(kv)
    return manager, kv


@pytest.fixture
def reconstructor(compressed_manager):
    manager, _ = compressed_manager
    return KVReconstructor(manager, target_dtype=torch.float16)


class TestSingleTokenReconstruction:

    def test_anchor_token_exact(self, compressed_manager, reconstructor):
        """Anchor tokens must reconstruct exactly (no quantization)."""
        manager, kv = compressed_manager
        for anchor_idx in list(manager.anchor_table.keys())[:5]:
            recon = reconstructor.reconstruct_token(anchor_idx)
            original = kv[anchor_idx]
            assert torch.allclose(recon, original, atol=1e-3), \
                f"Anchor {anchor_idx} not reconstructed exactly"

    def test_delta_token_error_bounded(self, compressed_manager, reconstructor):
        """Non-anchor tokens should reconstruct with low relative error."""
        manager, kv = compressed_manager
        delta_tokens = list(manager.delta_table.keys())[:20]
        for tok in delta_tokens:
            recon = reconstructor.reconstruct_token(tok)
            original = kv[tok].float()
            recon_f  = recon.float()
            rel_err  = (original - recon_f).norm() / (original.norm() + 1e-9)
            assert rel_err.item() < 0.05, \
                f"Token {tok} relative error too large: {rel_err:.4f}"

    def test_output_dtype_fp16(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        recon = reconstructor.reconstruct_token(0)
        assert recon.dtype == torch.float16

    def test_output_shape(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        recon = reconstructor.reconstruct_token(10)
        assert recon.shape == (2, HEADS, HEAD_DIM)

    def test_missing_delta_raises(self, compressed_manager):
        manager, _ = compressed_manager
        recon = KVReconstructor(manager)
        # Manually remove a delta entry to simulate corruption
        delta_tok = next(iter(manager.delta_table.keys()))
        del manager.delta_table[delta_tok]
        with pytest.raises(KeyError):
            recon.reconstruct_token(delta_tok)


class TestGroupedReconstruction:

    def test_range_output_shape(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        result = reconstructor.reconstruct_range(0, 63)
        assert result.kv.shape == (64, 2, HEADS, HEAD_DIM)

    def test_range_covers_multiple_anchor_segments(self, compressed_manager, reconstructor):
        """Range spanning 3 anchor segments should still reconstruct correctly."""
        manager, kv = compressed_manager
        result = reconstructor.reconstruct_range(0, 95)   # 0-31, 32-63, 64-95
        assert result.anchor_loads >= 3   # should load 3 distinct anchors

    def test_range_error_bounded(self, compressed_manager, reconstructor):
        manager, kv = compressed_manager
        result = reconstructor.reconstruct_range(0, 63)
        recon_f = result.kv.float()
        orig_f  = kv[0:64].float()
        rel_err = (orig_f - recon_f).norm() / (orig_f.norm() + 1e-9)
        assert rel_err.item() < 0.05, f"Range recon error: {rel_err:.4f}"

    def test_single_token_range(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        result = reconstructor.reconstruct_range(5, 5)
        assert result.kv.shape == (1, 2, HEADS, HEAD_DIM)
        assert result.num_tokens == 1

    def test_first_token_range(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        result = reconstructor.reconstruct_range(0, 0)
        assert result.kv.shape == (1, 2, HEADS, HEAD_DIM)
        # Token 0 is anchor — should be exact
        assert torch.allclose(result.kv, kv[0:1], atol=1e-3)

    def test_full_sequence_range(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        result = reconstructor.reconstruct_range(0, SEQ_LEN - 1)
        assert result.kv.shape == (SEQ_LEN, 2, HEADS, HEAD_DIM)

    def test_result_metadata(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        result = reconstructor.reconstruct_range(32, 63)
        assert result.token_start == 32
        assert result.token_end   == 63
        assert result.elapsed_ms  >= 0.0
        assert result.bytes_read  > 0

    def test_grouped_vs_single_token_consistency(self, compressed_manager, reconstructor):
        """Grouped reconstruction should match single-token results."""
        manager, kv = compressed_manager
        start, end = 32, 47

        grouped = reconstructor.reconstruct_range(start, end)
        for i, tok in enumerate(range(start, end + 1)):
            single = reconstructor.reconstruct_token(tok)
            assert torch.allclose(grouped.kv[i], single, atol=1e-4), \
                f"Mismatch at token {tok}"


class TestMeasureError:

    def test_measure_error_returns_correct_keys(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        err = reconstructor.measure_error(kv, 0, 31)
        assert set(err.keys()) == {"mean_l2", "max_l2", "mean_relative", "max_relative"}

    def test_anchor_only_range_has_low_error(self, compressed_manager, reconstructor):
        """Range containing only an anchor token should have near-zero error."""
        _, kv = compressed_manager
        err = reconstructor.measure_error(kv, 0, 0)
        assert err["mean_relative"] < 1e-4

    def test_error_metrics_are_non_negative(self, compressed_manager, reconstructor):
        _, kv = compressed_manager
        err = reconstructor.measure_error(kv, 0, 63)
        for k, v in err.items():
            assert v >= 0.0, f"Negative error metric: {k}={v}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
tests/test_compression.py

Unit tests for the compression layer:
  - quantize_int8 / dequantize_int8 round-trip
  - DeltaEncoder encode/decode correctness
  - QuantizedDelta byte size calculations
  - Near-zero delta edge case
  - Large magnitude delta edge case
"""

import sys
import math
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.quantization import quantize_int8, dequantize_int8, QuantizedDelta
from compression.delta_encoder import DeltaEncoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_kv():
    torch.manual_seed(0)
    return torch.randn(2, 32, 128, dtype=torch.float32)

@pytest.fixture
def zero_kv():
    return torch.zeros(2, 32, 128, dtype=torch.float32)

@pytest.fixture
def large_kv():
    torch.manual_seed(1)
    return torch.randn(2, 32, 128, dtype=torch.float32) * 100.0

@pytest.fixture
def fp16_kv():
    torch.manual_seed(2)
    return torch.randn(2, 32, 128, dtype=torch.float16)


# ── quantize_int8 / dequantize_int8 ──────────────────────────────────────────

class TestQuantizationRoundTrip:

    def test_basic_round_trip_error_bounded(self, random_kv):
        """Round-trip error should be small relative to input magnitude."""
        q = quantize_int8(random_kv)
        recon = dequantize_int8(q, target_dtype=torch.float32)
        rel_err = (random_kv - recon).norm() / (random_kv.norm() + 1e-9)
        # INT8 should stay within ~1% relative error for normal data
        assert rel_err.item() < 0.02, f"Relative error too large: {rel_err:.4f}"

    def test_output_dtype_fp16(self, random_kv):
        q = quantize_int8(random_kv)
        recon = dequantize_int8(q, target_dtype=torch.float16)
        assert recon.dtype == torch.float16

    def test_output_shape_preserved(self, random_kv):
        q = quantize_int8(random_kv)
        recon = dequantize_int8(q, target_dtype=torch.float32)
        assert recon.shape == random_kv.shape

    def test_zero_input(self, zero_kv):
        """Zero delta should quantize to all zeros without divide-by-zero."""
        q = quantize_int8(zero_kv)
        recon = dequantize_int8(q, target_dtype=torch.float32)
        assert torch.allclose(recon, zero_kv, atol=1e-6)

    def test_large_magnitude(self, large_kv):
        """Large magnitude inputs should still round-trip correctly."""
        q = quantize_int8(large_kv)
        recon = dequantize_int8(q, target_dtype=torch.float32)
        rel_err = (large_kv - recon).norm() / (large_kv.norm() + 1e-9)
        assert rel_err.item() < 0.02

    def test_quantized_dtype_is_int8(self, random_kv):
        q = quantize_int8(random_kv)
        assert q.data.dtype == torch.int8

    def test_scale_is_positive(self, random_kv):
        q = quantize_int8(random_kv)
        assert q.scale > 0.0

    def test_byte_size_calculation(self, random_kv):
        q = quantize_int8(random_kv)
        expected = random_kv.numel() * 1 + 4   # 1 byte/elem + 4-byte scale
        assert q.nbytes() == expected

    def test_compression_ratio_vs_fp16(self, random_kv):
        q = quantize_int8(random_kv)
        ratio = q.compression_ratio_vs_fp16()
        # Should be close to 2x (FP16=2bytes, INT8=1byte)
        assert 1.9 < ratio < 2.1, f"Expected ~2x ratio, got {ratio:.3f}"

    def test_shape_stored_correctly(self, random_kv):
        q = quantize_int8(random_kv)
        assert q.shape == tuple(random_kv.shape)


# ── DeltaEncoder ─────────────────────────────────────────────────────────────

class TestDeltaEncoder:

    def setup_method(self):
        self.encoder = DeltaEncoder()

    def test_encode_decode_round_trip(self, fp16_kv):
        torch.manual_seed(3)
        anchor = torch.randn_like(fp16_kv)
        q = self.encoder.encode(fp16_kv, anchor)
        recon = self.encoder.decode(anchor, q, dtype=torch.float16)
        rel_err = self.encoder.reconstruction_error(fp16_kv, recon)
        assert rel_err < 0.02, f"DeltaEncoder round-trip error: {rel_err:.4f}"

    def test_encode_identical_kv_gives_zero_delta(self):
        torch.manual_seed(4)
        kv = torch.randn(2, 16, 64, dtype=torch.float16)
        # If kv == anchor, delta should be near zero
        q = self.encoder.encode(kv, kv)
        recon = self.encoder.decode(kv, q, dtype=torch.float16)
        err = self.encoder.absolute_reconstruction_error(kv, recon)
        assert err < 1e-2, f"Zero-delta error too large: {err}"

    def test_encode_sequence_produces_correct_tables(self):
        torch.manual_seed(5)
        seq_len = 64
        kv_seq = torch.randn(seq_len, 2, 8, 32, dtype=torch.float16)
        anchors = [0, 16, 32, 48]
        anchor_table, delta_table = self.encoder.encode_sequence(kv_seq, anchors)

        assert set(anchor_table.keys()) == set(anchors)
        expected_delta_keys = set(range(seq_len)) - set(anchors)
        assert set(delta_table.keys()) == expected_delta_keys

    def test_encode_sequence_first_token_must_be_anchor(self):
        """Token 0 MUST be in anchor_positions; otherwise raises AssertionError."""
        torch.manual_seed(6)
        kv_seq = torch.randn(32, 2, 8, 32, dtype=torch.float16)
        with pytest.raises(AssertionError):
            self.encoder.encode_sequence(kv_seq, anchor_positions=[16])

    def test_reconstruction_error_zero_for_perfect_recon(self):
        x = torch.randn(2, 8, 32, dtype=torch.float16)
        err = self.encoder.reconstruction_error(x, x)
        assert err < 1e-6

    def test_absolute_error_nonzero_for_noisy_recon(self):
        x = torch.randn(2, 8, 32, dtype=torch.float16)
        y = x + 0.1
        err = self.encoder.absolute_reconstruction_error(x, y)
        assert err > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

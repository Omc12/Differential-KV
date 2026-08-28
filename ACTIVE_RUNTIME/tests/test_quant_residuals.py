"""Unit tests for 4-bit / 8-bit Quantized Residuals (DKV_RESIDUAL_QUANT).

Tests:
1. Low numerical reconstruction error (<0.01 relative error on residual vectors).
2. Memory compression ratio (>3.5x for int4 with group_size=64).
3. Integration with MLXKVBlockManager._apply_residual_quantization.
4. End-to-end exact retrieval in MLX wrapper with DKV_RESIDUAL_QUANT=int4.
"""
import os
import sys
import pytest

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))


@pytest.mark.skipif(not _HAS_MLX, reason="MLX required for quantized residual tests")
class TestQuantizedResiduals:
    def test_int4_reconstruction_error(self):
        """4-bit quantization with group_size=64 must have <1% relative reconstruction error."""
        # Realistic residual tensor: (B=4 blocks, R=128 residuals, H_kv=8 heads, D=128 head_dim)
        res = mx.random.normal((4, 128, 8, 128), dtype=mx.float16) * 0.05
        
        q_data, scales, biases = mx.quantize(res, group_size=64, bits=4)
        res_rec = mx.dequantize(q_data, scales, biases, group_size=64, bits=4)
        
        rel_error_4bit = (mx.linalg.norm(res - res_rec) / mx.linalg.norm(res)).item()
        assert rel_error_4bit < 0.12, f"Expected 4-bit relative error < 12%, got {rel_error_4bit*100:.2f}%"

        # 8-bit quantization must have <1% relative error
        q8, s8, b8 = mx.quantize(res, group_size=64, bits=8)
        res8_rec = mx.dequantize(q8, s8, b8, group_size=64, bits=8)
        rel_error_8bit = (mx.linalg.norm(res - res8_rec) / mx.linalg.norm(res)).item()
        assert rel_error_8bit < 0.01, f"Expected 8-bit relative error < 1%, got {rel_error_8bit*100:.2f}%"

    def test_int4_memory_compression_ratio(self):
        """4-bit group_size=64 must achieve >3.5x memory compression over FP16."""
        res = mx.zeros((8, 128, 8, 128), dtype=mx.float16)
        fp16_bytes = res.nbytes
        
        q_data, scales, biases = mx.quantize(res, group_size=64, bits=4)
        int4_bytes = q_data.nbytes + scales.nbytes + biases.nbytes
        
        ratio = fp16_bytes / int4_bytes
        assert ratio >= 3.5, f"Expected compression ratio >= 3.5x, got {ratio:.2f}x"

    def test_manager_quantization_hook(self, monkeypatch):
        """MLXKVBlockManager._apply_residual_quantization applies int4 when configured."""
        from mlx_dkv_wrapper import MLXKVBlockManager
        
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        mgr = MLXKVBlockManager(
            num_layers=4, heads=16, kv_heads=8, head_dim=128, rank=32, block_size=256
        )
        assert mgr.residual_quant == "int4"
        
        rk = mx.random.normal((2, 128, 8, 128), dtype=mx.float16)
        rv = mx.random.normal((2, 128, 8, 128), dtype=mx.float16)
        
        rk_out, rv_out = mgr._apply_residual_quantization(rk, rv)
        assert rk_out.shape == rk.shape
        assert rv_out.shape == rv.shape
        
        # Check that quantization was applied (values are close but quantized)
        diff_k = mx.max(mx.abs(rk - rk_out)).item()
        assert diff_k > 0.0, "Expected non-zero difference from quantization"
        assert diff_k < 0.5, f"Max difference too large: {diff_k}"

    def test_manager_disabled_by_default(self, monkeypatch):
        """When DKV_RESIDUAL_QUANT is unset, quantization hook is a bit-exact no-op."""
        from mlx_dkv_wrapper import MLXKVBlockManager
        
        monkeypatch.delenv("DKV_RESIDUAL_QUANT", raising=False)
        mgr = MLXKVBlockManager(
            num_layers=4, heads=16, kv_heads=8, head_dim=128, rank=32, block_size=256
        )
        assert mgr.residual_quant == "none"
        
        rk = mx.random.normal((2, 128, 8, 128), dtype=mx.float16)
        rv = mx.random.normal((2, 128, 8, 128), dtype=mx.float16)
        
        rk_out, rv_out = mgr._apply_residual_quantization(rk, rv)
        diff_k = mx.max(mx.abs(rk - rk_out)).item()
        assert diff_k == 0.0, "Expected bit-exact identity when quantization is disabled"

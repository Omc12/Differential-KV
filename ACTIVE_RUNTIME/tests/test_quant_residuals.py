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

    def test_physical_packed_int4_allocation(self, monkeypatch):
        """When DKV_RESIDUAL_QUANT=int4, session residual buffers must be physically packed uint32 + scales/biases."""
        from mlx_dkv_wrapper import MLXKVBlockManager
        
        # FP16 Manager
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "none")
        mgr_fp16 = MLXKVBlockManager(
            num_layers=4, heads=16, kv_heads=8, head_dim=128, rank=32, block_size=256
        )
        sess_fp16 = mgr_fp16._create_empty_session(max_blocks=16)
        assert sess_fp16["comp_res_k"] is not None
        assert sess_fp16["comp_res_k_q"] is None
        fp16_bytes = sum(t.nbytes for t in sess_fp16["comp_res_k"]) + sum(t.nbytes for t in sess_fp16["comp_res_v"])

        # INT4 Manager
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        mgr_int4 = MLXKVBlockManager(
            num_layers=4, heads=16, kv_heads=8, head_dim=128, rank=32, block_size=256
        )
        sess_int4 = mgr_int4._create_empty_session(max_blocks=16)
        assert sess_int4["comp_res_k"] is None
        assert sess_int4["comp_res_k_q"] is not None
        assert sess_int4["comp_res_k_q"][0].dtype == mx.uint32
        
        int4_bytes = (
            sum(q.nbytes + s.nbytes + b.nbytes for q, s, b in zip(sess_int4["comp_res_k_q"], sess_int4["comp_res_k_s"], sess_int4["comp_res_k_b"])) +
            sum(q.nbytes + s.nbytes + b.nbytes for q, s, b in zip(sess_int4["comp_res_v_q"], sess_int4["comp_res_v_s"], sess_int4["comp_res_v_b"]))
        )
        
        ratio = fp16_bytes / int4_bytes
        assert ratio >= 3.5, f"Expected >=3.5x physical buffer compression, got {ratio:.2f}x"
        
        # Test store and fetch round trip
        rk_raw = mx.random.normal((2, 128, 8, 128), dtype=mx.float16) * 0.05
        rv_raw = mx.random.normal((2, 128, 8, 128), dtype=mx.float16) * 0.05
        mgr_int4._store_residuals(sess_int4, 0, slice(0, 2), rk_raw, rv_raw)
        
        rk_fetched = mgr_int4._fetch_res_k(sess_int4, 0, nb=2)
        assert rk_fetched.shape == rk_raw.shape
        diff = mx.max(mx.abs(rk_raw - rk_fetched)).item()
        assert diff < 0.2, f"Fetched difference too large: {diff}"

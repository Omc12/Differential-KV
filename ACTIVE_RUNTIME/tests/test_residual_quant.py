"""Unit tests for Group-Quantized Residual Buffers (INT4 / INT8).

Validates:
1. Numerical accuracy and bounds across random vectors.
2. Bit-packing exactness for INT4 (8 elems/int32) and INT8 (4 elems/int32).
3. Group sizes: group_size=64 across head_dim=128 and head_dim=256.
4. Physical memory reduction: verify 3.56x smaller buffer allocation.
5. CUDA memory allocator verification via torch.cuda.memory_allocated().
"""

import math
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.compression.residual_quant import (
    quantize_residuals_group_asymmetric,
    dequantize_residuals_group_asymmetric,
    get_packed_width,
)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_quant_dequant_roundtrip(head_dim, bits):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    group_size = 64
    x = torch.randn(8, 32, 4, head_dim, device=device, dtype=torch.float16)

    packed_q, scale, bias = quantize_residuals_group_asymmetric(
        x, group_size=group_size, bits=bits
    )

    expected_width = (head_dim * bits + 31) // 32
    assert packed_q.shape == (8, 32, 4, expected_width)
    assert scale.shape == (8, 32, 4, head_dim // group_size)
    assert bias.shape == (8, 32, 4, head_dim // group_size)

    deq = dequantize_residuals_group_asymmetric(
        packed_q, scale, bias, group_size=group_size, bits=bits, head_dim=head_dim
    )
    assert deq.shape == x.shape
    assert deq.dtype == x.dtype

    # Quantization error should be bounded by group step size
    max_err = (x - deq).abs().max().item()
    assert max_err < 0.5, f"Max error {max_err} exceeded expected quantization tolerance"


def test_physical_buffer_memory_ratio():
    """Verify the 3.56x physical memory reduction specification."""
    num_blocks = 512
    max_residual = 128
    kv_heads = 4
    head_dim = 128
    group_size = 64
    bits = 4

    # FP16 buffer bytes
    fp16_elements = num_blocks * max_residual * kv_heads * head_dim
    fp16_bytes = fp16_elements * 2  # float16 = 2 bytes

    # INT4 buffer bytes
    packed_width = (head_dim * bits + 31) // 32
    num_groups = head_dim // group_size
    packed_elements = num_blocks * max_residual * kv_heads * packed_width
    packed_bytes = packed_elements * 4  # int32 = 4 bytes
    scale_bytes = (num_blocks * max_residual * kv_heads * num_groups) * 2
    bias_bytes = (num_blocks * max_residual * kv_heads * num_groups) * 2
    int4_total_bytes = packed_bytes + scale_bytes + bias_bytes

    ratio = fp16_bytes / int4_total_bytes
    # 256 / 72 = 3.5555...
    assert math.isclose(ratio, 3.555555555, rel_tol=1e-3)
    assert 3.55 <= ratio <= 3.57


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory allocator check")
def test_cuda_allocator_actual_memory():
    """Verify actual CUDA allocator physical memory reduction."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    num_blocks = 256
    max_residual = 128
    kv_heads = 4
    head_dim = 128
    group_size = 64

    # 1. Allocate FP16 buffers
    fp16_k = torch.zeros(num_blocks, max_residual, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    fp16_v = torch.zeros(num_blocks, max_residual, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    fp16_mem = torch.cuda.memory_allocated() - base_mem

    del fp16_k, fp16_v
    torch.cuda.empty_cache()

    # 2. Allocate INT4 buffers
    packed_width = get_packed_width(head_dim, 4)
    num_groups = head_dim // group_size
    int4_k_q = torch.zeros(num_blocks, max_residual, kv_heads, packed_width, device="cuda", dtype=torch.int32)
    int4_k_s = torch.zeros(num_blocks, max_residual, kv_heads, num_groups, device="cuda", dtype=torch.float16)
    int4_k_b = torch.zeros(num_blocks, max_residual, kv_heads, num_groups, device="cuda", dtype=torch.float16)
    int4_v_q = torch.zeros(num_blocks, max_residual, kv_heads, packed_width, device="cuda", dtype=torch.int32)
    int4_v_s = torch.zeros(num_blocks, max_residual, kv_heads, num_groups, device="cuda", dtype=torch.float16)
    int4_v_b = torch.zeros(num_blocks, max_residual, kv_heads, num_groups, device="cuda", dtype=torch.float16)
    int4_mem = torch.cuda.memory_allocated() - base_mem

    measured_ratio = fp16_mem / int4_mem
    del int4_k_q, int4_k_s, int4_k_b, int4_v_q, int4_v_s, int4_v_b
    torch.cuda.empty_cache()

    assert math.isclose(measured_ratio, 3.5555555, rel_tol=1e-2), (
        f"Measured allocator memory ratio was {measured_ratio:.3f}, expected ~3.56"
    )


class TestNativeBlockPoolQuantization:
    def test_pool_quantized_allocation(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        from runtime.native_block_pool import NativeBlockPool
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pool = NativeBlockPool(
            max_blocks=128,
            num_kv_heads=4,
            head_dim=128,
            rank=32,
            max_seq_len=64,
            device=dev,
            dtype=torch.float16,
            initial_blocks=64,
            max_residual_tokens=64,
        )
        assert pool.residual_quant == "int4"
        # Zero FP16 bytes stored in persistent buffer
        assert pool._residual_K_values is None
        assert pool._residual_V_values is None
        assert pool.comp_res_k_q is not None
        assert pool.comp_res_k_q.dtype == torch.int32
        assert pool.comp_res_k_s is not None
        assert pool.comp_res_k_s.dtype == torch.float16

    def test_pool_write_and_gather_parity(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        from runtime.native_block_pool import NativeBlockPool
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pool = NativeBlockPool(
            max_blocks=128,
            num_kv_heads=2,
            head_dim=128,
            rank=16,
            max_seq_len=64,
            device=dev,
            dtype=torch.float16,
            initial_blocks=64,
            max_residual_tokens=32,
        )
        pidx = pool.allocate_block()
        U = torch.randn(64, 16, device=dev, dtype=torch.float16)
        V = torch.randn(16, 2 * 2 * 128, device=dev, dtype=torch.float16)
        anc_k = torch.randn(2, 128, device=dev, dtype=torch.float16)
        anc_v = torch.randn(2, 128, device=dev, dtype=torch.float16)
        res_pos = torch.arange(10, device=dev, dtype=torch.int16)
        res_val_k = torch.randn(10, 2, 128, device=dev, dtype=torch.float16)
        res_val_v = torch.randn(10, 2, 128, device=dev, dtype=torch.float16)

        pool.write_block(
            pool_idx=pidx,
            U=U, V=V,
            anchor_K=anc_k, anchor_V=anc_v,
            scale=1.0, seq_len=64,
            residual_K_positions=res_pos,
            residual_K_values=res_val_k,
            residual_V_positions=res_pos,
            residual_V_values=res_val_v
        )

        # Test single block gather via unified get_residual_k / get_residual_v
        recon_k = pool.get_residual_k(pidx)
        recon_v = pool.get_residual_v(pidx)
        assert recon_k.shape == (32, 2, 128)
        assert recon_v.shape == (32, 2, 128)

        # Check numerical error on the 10 written residual rows
        err_k = (recon_k[:10] - res_val_k).abs().max().item()
        err_v = (recon_v[:10] - res_val_v).abs().max().item()
        assert err_k < 0.5, f"Reconstructed K residual error {err_k} exceeded tolerance"
        assert err_v < 0.5, f"Reconstructed V residual error {err_v} exceeded tolerance"

    def test_pool_batched_write_and_grow(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        from runtime.native_block_pool import NativeBlockPool
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pool = NativeBlockPool(
            max_blocks=256,
            num_kv_heads=2,
            head_dim=128,
            rank=16,
            max_seq_len=64,
            device=dev,
            dtype=torch.float16,
            initial_blocks=32,
            max_residual_tokens=32,
        )
        # Allocate 8 blocks
        pidx = torch.tensor([pool.allocate_block() for _ in range(8)], device=dev, dtype=torch.long)
        N = 8
        U = torch.randn(N, 64, 16, device=dev, dtype=torch.float16)
        V = torch.randn(N, 16, 2 * 2 * 128, device=dev, dtype=torch.float16)
        anc_k = torch.randn(N, 2, 128, device=dev, dtype=torch.float16)
        anc_v = torch.randn(N, 2, 128, device=dev, dtype=torch.float16)
        scales = torch.ones(N, device=dev, dtype=torch.float16)
        res_pos = torch.arange(8, device=dev, dtype=torch.int16).unsqueeze(0).expand(N, -1)
        res_val_k = torch.randn(N, 8, 2, 128, device=dev, dtype=torch.float16)
        res_val_v = torch.randn(N, 8, 2, 128, device=dev, dtype=torch.float16)

        pool.write_blocks_batched(
            pool_indices=pidx,
            U=U, V=V,
            anchor_K=anc_k, anchor_V=anc_v,
            scales=scales, seq_len=64,
            res_K_positions=res_pos,
            res_K_values=res_val_k,
            res_V_positions=res_pos,
            res_V_values=res_val_v
        )

        # Gather batch of indices
        gather_indices = pidx[:4]
        g_k = pool.get_residual_k(gather_indices)
        g_v = pool.get_residual_v(gather_indices)
        assert g_k.shape == (4, 32, 2, 128)
        assert g_v.shape == (4, 32, 2, 128)

        # Test growing the pool
        old_cap = pool.current_blocks
        pool._grow_pool(old_cap + 64)
        assert pool.current_blocks > old_cap


        # Verify data preserved after grow
        g_k_after = pool.get_residual_k(gather_indices)
        assert (g_k - g_k_after).abs().max().item() == 0.0


class TestResidualFormatIsNotAnAlias:
    """int8 must not be a silent alias for int4, and the format must not be
    decided by the environment alone.

    Both defects were real and neither was visible from outside the allocator.
    Until e38f3cd1 the bit width came from DKV_RESIDUAL_QUANT_BITS (default 4)
    regardless of the format NAME, so "int8" allocated a 4-bit packed_width and
    produced byte-identical buffers with identical error -- and DKVConfig's
    residual_quant was dead config the pool never saw, so a caller that passed a
    config object got whatever the environment said instead.
    """

    def _pool(self, **kw):
        from runtime.native_block_pool import NativeBlockPool
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        base = dict(max_blocks=64, num_kv_heads=4, head_dim=128, rank=16,
                    max_seq_len=64, device=dev, dtype=torch.float16,
                    initial_blocks=32, max_residual_tokens=32)
        base.update(kw)
        return NativeBlockPool(**base)

    def test_int8_allocates_twice_int4(self, monkeypatch):
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        monkeypatch.delenv("DKV_RESIDUAL_QUANT_BITS", raising=False)
        p4 = self._pool()
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int8")
        p8 = self._pool()

        assert (p4.residual_quant_bits, p8.residual_quant_bits) == (4, 8)
        # packed_width, not just a flag: this is the number that was wrong.
        assert p4.comp_res_k_q.shape[-1] == 16
        assert p8.comp_res_k_q.shape[-1] == 32
        assert p8.comp_res_k_q.nbytes == 2 * p4.comp_res_k_q.nbytes
        assert p8.comp_res_v_q.nbytes == 2 * p4.comp_res_v_q.nbytes

    def test_explicit_bits_still_override_the_name(self, monkeypatch):
        # Sweeps that vary width without renaming the format must keep working.
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int8")
        monkeypatch.setenv("DKV_RESIDUAL_QUANT_BITS", "4")
        p = self._pool()
        assert p.residual_quant == "int8" and p.residual_quant_bits == 4
        assert p.comp_res_k_q.shape[-1] == 16

    def test_constructor_beats_environment(self, monkeypatch):
        # KVRuntimeManager forwards DKVConfig's value here; if the env won, the
        # config object would be dead again.
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int4")
        monkeypatch.delenv("DKV_RESIDUAL_QUANT_BITS", raising=False)
        p = self._pool(residual_quant="int8")
        assert p.residual_quant == "int8" and p.residual_quant_bits == 8
        assert p.comp_res_k_q.shape[-1] == 32

    def test_unsupported_bit_width_falls_back_to_the_name(self, monkeypatch):
        # residual_quant.py only has shift tables for 4 and 8; a bogus width has
        # to be caught at ALLOCATION, not at the first write.
        monkeypatch.setenv("DKV_RESIDUAL_QUANT", "int8")
        monkeypatch.setenv("DKV_RESIDUAL_QUANT_BITS", "5")
        p = self._pool()
        assert p.residual_quant_bits == 8

    def test_config_default_matches_the_serving_default(self, monkeypatch):
        """One dial, two live defaults -- they must agree.

        DKVConfig's default is what every direct-construction path allocates
        (serving/hf_dkv_wrapper.py, and so colab/run_nat_eval.py, which declines
        apply_best_decode_defaults on purpose). BEST_DECODE_DEFAULTS is what the
        CLI and gateway get. They drifted apart once already, which is how int4
        shipped to one set of callers and not the other.
        """
        # monkeypatch, not os.environ.pop: this asserts on the DEFAULT, so it
        # has to clear the env, and clearing it for real would leak into every
        # test that runs after this one.
        for k in ("DKV_RESIDUAL_QUANT", "DKV_RESIDUAL_QUANT_BITS"):
            monkeypatch.delenv(k, raising=False)
        from native_core.config import DKVConfig
        from serving.decode_config import BEST_DECODE_DEFAULTS
        cfg = DKVConfig({})
        assert cfg.residual_quant == BEST_DECODE_DEFAULTS["DKV_RESIDUAL_QUANT"]
        # And the width follows the name without anyone setting BITS.
        assert cfg.residual_quant_bits == (8 if "8" in cfg.residual_quant else 4)

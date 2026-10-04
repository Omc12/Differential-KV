"""Hybrid store (DKV_KEY_QUANT): per-channel quantized keys in the pool.

Checks, from the storage format outwards:
  1. quantize/dequantize round trip matches the reference per-channel
     quantizer (the one the tier-1/tier-2 measurements used) and its error
     bound, for bits 4 and 8, including a sequence that is not a multiple of
     the group size;
  2. the pool allocates, writes, grows and reads the codes, and the slot reads
     back what was written; with the switch off nothing exists;
  3. reconstruct_blocks uses k_delta for keys and the factor for values;
  4. history attention with key_deltas equals the same attention with a factor
     that reproduces those deltas exactly.
"""
import os
import importlib

import pytest
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ref_quant(x, bits, group=32):
    """Reference: the tier-1 per-channel quantizer (float scales)."""
    N, S, H, D = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    q = float(2 ** bits - 1)
    for g0 in range(0, S, group):
        blk = x[:, g0:g0 + group].float()
        lo = blk.amin(1, keepdim=True)
        hi = blk.amax(1, keepdim=True)
        s = ((hi - lo).clamp(min=1e-8) / q).half().float()
        z = lo.half().float()
        out[:, g0:g0 + group] = torch.round((blk - z) / s).clamp(0, q) * s + z
    return out


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("S", [64, 1000])
def test_roundtrip_matches_reference(bits, S):
    from runtime.native_block_pool import (quantize_keys_per_channel,
                                           dequantize_keys_per_channel)
    torch.manual_seed(0)
    x = torch.randn(3, S, 4, 64, device=DEV) * torch.linspace(0.1, 8.0, 64, device=DEV)
    codes, scale, zero = quantize_keys_per_channel(x, bits, 32)
    assert codes.dtype == torch.uint8
    assert codes.shape[-1] == (32 if bits == 4 else 64)
    y = dequantize_keys_per_channel(codes, scale, zero, bits, 32, torch.float32)
    ref = _ref_quant(x, bits, 32)
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()
    # error bound: half a step per channel group (plus fp16 rounding of z)
    step = (scale.float().repeat_interleave(32, 1)[:, :S])
    assert ((y - x).abs() <= step * 0.5 + 1e-2).all()


def _make_pool(monkeypatch, kq):
    if kq:
        monkeypatch.setenv("DKV_KEY_QUANT", kq)
    else:
        monkeypatch.delenv("DKV_KEY_QUANT", raising=False)
    monkeypatch.setenv("DKV_DISABLE_CUDA_GRAPH", "1")
    import runtime.native_block_pool as nbp
    importlib.reload(nbp)
    return nbp, nbp.NativeBlockPool(max_blocks=64, num_kv_heads=2, head_dim=64, rank=8,
                                    max_seq_len=96, device=DEV, initial_blocks=4,
                                    num_layers=1, lazy=False, max_residual_tokens=8)


def test_pool_off_has_no_codes(monkeypatch):
    _, pool = _make_pool(monkeypatch, "")
    assert pool.key_quant_bits == 0 and pool.kq_codes is None
    assert pool.get_key_deltas(torch.tensor([0], device=DEV)) is None


def test_pool_write_read_and_grow(monkeypatch):
    nbp, pool = _make_pool(monkeypatch, "pc4")
    assert pool.key_quant_bits == 4 and pool.kq_codes is not None
    N, S, H, D, R = 3, 90, 2, 64, 8
    torch.manual_seed(1)
    kd = torch.randn(N, S, H, D, device=DEV)
    slots = torch.tensor(pool.allocate_blocks(N), device=DEV)
    pool.write_blocks_batched(
        pool_indices=slots, U=torch.randn(N, S, R, device=DEV),
        V=torch.randn(N, R, 2 * H * D, device=DEV),
        anchor_K=torch.randn(N, H, D, device=DEV), anchor_V=torch.randn(N, H, D, device=DEV),
        scales=torch.ones(N, device=DEV), seq_len=S, key_deltas=kd)
    got = pool.get_key_deltas(slots).float()[:, :S]   # slots are max_seq_len wide
    ref = _ref_quant(kd, 4, 32)
    assert torch.allclose(got, ref, atol=2e-3), (got - ref).abs().max()
    # grow keeps written slots intact
    before = pool.kq_codes[slots].clone()
    pool._grow_pool(pool.current_blocks + 8)
    assert torch.equal(pool.kq_codes[slots], before)
    # the per-block path must refuse rather than store a slot without keys
    with pytest.raises(RuntimeError):
        pool.write_block(int(slots[0]), torch.zeros(S, R, device=DEV),
                         torch.zeros(R, 2 * H * D, device=DEV),
                         torch.zeros(H, D, device=DEV), torch.zeros(H, D, device=DEV),
                         1.0, S)
    # missing key_deltas on the batched path is an error, not a silent zero
    with pytest.raises(ValueError):
        pool.write_blocks_batched(
            pool_indices=slots[:1], U=torch.randn(1, S, R, device=DEV),
            V=torch.randn(1, R, 2 * H * D, device=DEV),
            anchor_K=torch.randn(1, H, D, device=DEV), anchor_V=torch.randn(1, H, D, device=DEV),
            scales=torch.ones(1, device=DEV), seq_len=S)
    monkeypatch.delenv("DKV_KEY_QUANT", raising=False)
    importlib.reload(nbp)


@pytest.mark.parametrize("rq", ["int8", "none"])
def test_trimmed_store(monkeypatch, rq):
    """Value-only factor, and K residuals served from the codes."""
    monkeypatch.setenv("DKV_KEY_QUANT", "pc4")
    monkeypatch.setenv("DKV_DISABLE_CUDA_GRAPH", "1")
    import runtime.native_block_pool as nbp
    importlib.reload(nbp)
    pool = nbp.NativeBlockPool(max_blocks=64, num_kv_heads=2, head_dim=64, rank=8,
                               max_seq_len=96, device=DEV, initial_blocks=4, num_layers=1,
                               lazy=False, max_residual_tokens=8, residual_quant=rq)
    assert pool.V_KV.shape[1] == 1                       # value half only
    assert pool.comp_res_k_q is None and pool._residual_K_values is None
    N, S, H, D, R = 2, 90, 2, 64, 8
    torch.manual_seed(3)
    kd = torch.randn(N, S, H, D, device=DEV)
    V = torch.randn(N, R, 2 * H * D, device=DEV)
    pos = torch.full((N, 8), -1, device=DEV, dtype=torch.int16)
    pos[:, :5] = torch.tensor([3, 40, 41, 77, 89], device=DEV, dtype=torch.int16)
    slots = torch.tensor(pool.allocate_blocks(N), device=DEV)
    pool.write_blocks_batched(
        pool_indices=slots, U=torch.randn(N, S, R, device=DEV), V=V,
        anchor_K=torch.randn(N, H, D, device=DEV), anchor_V=torch.randn(N, H, D, device=DEV),
        scales=torch.ones(N, device=DEV), seq_len=S, key_deltas=kd,
        res_K_positions=pos, res_K_values=torch.randn(N, 8, H, D, device=DEV),
        res_V_positions=pos, res_V_values=torch.randn(N, 8, H, D, device=DEV))
    assert torch.count_nonzero(pool.V_K[slots]) == 0      # broadcast zeros
    vv = V[:, :, H * D:].reshape(N, R, H, D).to(pool.dtype)
    assert torch.allclose(pool.V_V[slots].float(), vv.float())
    rk = pool.get_residual_k(slots).float()               # [N, 8, H, D]
    full = pool.get_key_deltas(slots).float()
    for n in range(N):
        for j in range(5):
            assert torch.allclose(rk[n, j], full[n, int(pos[n, j])], atol=1e-3)
        assert torch.count_nonzero(rk[n, 5:]) == 0        # padding reads zero
    pool._grow_pool(pool.current_blocks + 8)               # grows with K arrays absent
    assert torch.allclose(pool.get_residual_k(slots).float(), rk, atol=1e-6)
    monkeypatch.delenv("DKV_KEY_QUANT", raising=False)
    importlib.reload(nbp)


def test_reconstruct_uses_k_delta():
    from native_core.sparse_decode.remat_cache import reconstruct_blocks
    N, S, R, H, D = 2, 10, 4, 2, 8
    U = torch.randn(N, S, R, device=DEV)
    V_K = torch.randn(N, R, H, D, device=DEV)
    V_V = torch.randn(N, R, H, D, device=DEV)
    aK = torch.randn(N, H, D, device=DEV)
    aV = torch.randn(N, H, D, device=DEV)
    sc = torch.ones(N, device=DEV)
    us = torch.ones(N, device=DEV)
    kd = torch.randn(N, S, H, D, device=DEV)
    K, V = reconstruct_blocks(U, V_K, V_V, aK, aV, sc, us, R, k_delta=kd)
    assert torch.allclose(K[:, 1:].float(), (kd + aK.unsqueeze(1)).float(), atol=1e-5)
    K0, V0 = reconstruct_blocks(U, V_K, V_V, aK, aV, sc, us, R)
    assert torch.allclose(V.float(), V0.float(), atol=1e-5)      # values untouched
    assert torch.allclose(K[:, 0].float(), aK.float())            # anchor row first


def test_history_attention_key_deltas_equivalent():
    from native_core.sparse_decode.triton_fused_decode import (
        _prefill_fused_history_attend_compiled as f)
    torch.manual_seed(2)
    N, S, H, D, Q = 2, 12, 2, 8, 3
    kd = torch.randn(N, S, H, D, device=DEV)
    # a factor that reproduces kd exactly: U = I (R = S), V_K = kd
    U = torch.eye(S, device=DEV).unsqueeze(0).expand(N, S, S).contiguous()
    V_K = kd.clone()                                   # [N, R=S, H, D]
    V_V = torch.randn(N, S, H, D, device=DEV)
    aK, aV = torch.randn(N, H, D, device=DEV), torch.randn(N, H, D, device=DEV)
    cos = torch.ones(N, 1 + S, 1, D, device=DEV)
    sin = torch.zeros(N, 1 + S, 1, D, device=DEV)
    q = torch.randn(1, H, Q, D, device=DEV)
    sl = torch.full((N,), S, device=DEV, dtype=torch.int32)
    e = torch.empty(N, 0, device=DEV, dtype=torch.int16)
    ev = torch.empty(N, 0, H, D, device=DEV)
    common = dict(V_V=V_V, anchors_K=aK, anchors_V=aV, scales=torch.ones(N, device=DEV),
                  cos_sliced=cos, sin_sliced=sin, q=q, seq_lens=sl, inv_scale=D ** -0.5,
                  residual_K_positions=e, residual_K_values=ev,
                  residual_V_positions=e, residual_V_values=ev)
    ref = f(U=U, V_K=V_K, **common)
    got = f(U=U, V_K=torch.zeros_like(V_K), key_deltas=kd, **common)
    assert torch.allclose(got.float(), ref.float(), atol=1e-4), (got - ref).abs().max()

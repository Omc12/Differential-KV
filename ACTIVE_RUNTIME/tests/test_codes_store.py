"""Codes store (DKV_STORE=codes): per-channel key codes and per-token value
codes in the pool, KIVI's layout as a DKV store.

Checks, from the storage format outwards:
  1. 2-bit key codes round-trip against the reference per-channel quantizer;
  2. value codes round-trip against a reference per-token quantizer, at 2 and 4
     bits, within half a step;
  3. the pool allocates, writes, grows and reads both code arrays; the factor
     is rank 1, no value residual is stored, value residuals read zero;
  4. reconstruct_blocks takes values from v_delta;
  5. the fused history tile with value codes equals plain attention over the
     dequantized keys and values.
"""
import importlib

import pytest
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ref_value_quant(x, bits, group=32):
    N, S, H, D = x.shape
    xg = x.float().view(N, S, H, D // group, group)
    lo = xg.amin(-1, keepdim=True)
    hi = xg.amax(-1, keepdim=True)
    q = float(2 ** bits - 1)
    s = ((hi - lo).clamp(min=1e-8) / q).half().float()
    z = lo.half().float()
    return (torch.round((xg - z) / s).clamp(0, q) * s + z).view(N, S, H, D)


def test_key_codes_two_bit_roundtrip():
    from runtime.native_block_pool import (quantize_keys_per_channel,
                                           dequantize_keys_per_channel)
    torch.manual_seed(0)
    x = torch.randn(2, 100, 3, 64, device=DEV)
    codes, scale, zero = quantize_keys_per_channel(x, 2, 32)
    assert codes.shape[-1] == 16
    y = dequantize_keys_per_channel(codes, scale, zero, 2, 32, torch.float32)
    for g0 in range(0, 100, 32):
        blk = x[:, g0:g0 + 32].float()
        lo, hi = blk.amin(1, keepdim=True), blk.amax(1, keepdim=True)
        s = ((hi - lo).clamp(min=1e-8) / 3.0).half().float()
        z = lo.half().float()
        ref = torch.round((blk - z) / s).clamp(0, 3) * s + z
        assert torch.allclose(y[:, g0:g0 + 32], ref, atol=1e-5)


@pytest.mark.parametrize("bits", [2, 4])
def test_value_codes_roundtrip(bits):
    from runtime.native_block_pool import (quantize_values_per_token,
                                           dequantize_values_per_token)
    torch.manual_seed(1)
    x = torch.randn(2, 50, 3, 64, device=DEV) * 3
    codes, scale, zero = quantize_values_per_token(x, bits, 32)
    assert codes.shape == (2, 50, 3, 64 * bits // 8) and scale.shape == (2, 50, 3, 2)
    y = dequantize_values_per_token(codes, scale, zero, bits, 32, torch.float32)
    assert torch.allclose(y, _ref_value_quant(x, bits), atol=1e-4)
    step = scale.float().repeat_interleave(32, -1)
    assert ((y - x).abs() <= step * 0.5 + 1e-2).all()


def _codes_pool(monkeypatch, bits):
    monkeypatch.setenv("DKV_STORE", "codes")
    monkeypatch.setenv("DKV_QUANT_BITS", str(bits))
    monkeypatch.delenv("DKV_KEY_QUANT", raising=False)
    monkeypatch.delenv("DKV_VALUE_QUANT", raising=False)
    monkeypatch.setenv("DKV_DISABLE_CUDA_GRAPH", "1")
    import runtime.native_block_pool as nbp
    importlib.reload(nbp)
    return nbp, nbp.NativeBlockPool(max_blocks=64, num_kv_heads=2, head_dim=64, rank=8,
                                    max_seq_len=96, device=DEV, initial_blocks=4,
                                    num_layers=1, lazy=False, max_residual_tokens=8,
                                    residual_quant="int8")


@pytest.mark.parametrize("bits", [2, 4])
def test_pool_codes_write_read_grow(monkeypatch, bits):
    nbp, pool = _codes_pool(monkeypatch, bits)
    assert pool.key_quant_bits == bits and pool.value_quant_bits == bits
    assert pool.rank == 1 and pool.U.shape[2] == 1
    assert pool.comp_res_v_q is None and pool._residual_V_values is None
    N, S, H, D = 3, 90, 2, 64
    torch.manual_seed(2)
    kd = torch.randn(N, S, H, D, device=DEV)
    vd = torch.randn(N, S, H, D, device=DEV)
    pos = torch.full((N, 8), -1, device=DEV, dtype=torch.int16)
    pos[:, :3] = torch.tensor([5, 6, 70], device=DEV, dtype=torch.int16)
    slots = torch.tensor(pool.allocate_blocks(N), device=DEV)
    pool.write_blocks_batched(
        pool_indices=slots, U=torch.randn(N, S, 1, device=DEV),
        V=torch.randn(N, 1, 2 * H * D, device=DEV),
        anchor_K=torch.randn(N, H, D, device=DEV), anchor_V=torch.randn(N, H, D, device=DEV),
        scales=torch.ones(N, device=DEV), seq_len=S, key_deltas=kd, value_deltas=vd,
        res_K_positions=pos, res_K_values=torch.randn(N, 8, H, D, device=DEV),
        res_V_positions=pos, res_V_values=torch.randn(N, 8, H, D, device=DEV))
    got_v = pool.get_value_deltas(slots).float()[:, :S]
    assert torch.allclose(got_v, _ref_value_quant(vd, bits), atol=2e-3)
    assert (pool.residual_V_positions[slots] == -1).all()       # no value residual
    assert (pool.residual_K_positions[slots, :3] >= 0).all()    # router keeps its rows
    assert torch.count_nonzero(pool.get_residual_v(slots)) == 0
    before = (pool.kq_codes[slots].clone(), pool.vq_codes[slots].clone())
    pool._grow_pool(pool.current_blocks + 8)
    assert torch.equal(pool.kq_codes[slots], before[0])
    assert torch.equal(pool.vq_codes[slots], before[1])
    with pytest.raises(ValueError):                              # no silent zeros
        pool.write_blocks_batched(
            pool_indices=slots[:1], U=torch.randn(1, S, 1, device=DEV),
            V=torch.randn(1, 1, 2 * H * D, device=DEV),
            anchor_K=torch.randn(1, H, D, device=DEV), anchor_V=torch.randn(1, H, D, device=DEV),
            scales=torch.ones(1, device=DEV), seq_len=S, key_deltas=kd[:1])
    # bytes per block: codes and their scales, plus a small fixed remainder
    T = pool.max_seq_len
    codes_b = (T * H * D * bits // 8 * 2                 # key + value codes
               + 2 * ((T + 31) // 32) * H * D * 2        # key scale + zero
               + 2 * T * H * (D // 32) * 2)              # value scale + zero
    rest = pool._bytes_per_block - codes_b
    assert 0 < rest < T * 1 + H * D * 2 * 3 + 8 * 4 + 16, rest
    for v in ("DKV_STORE", "DKV_QUANT_BITS"):
        monkeypatch.delenv(v, raising=False)
    importlib.reload(nbp)


def test_reconstruct_uses_v_delta():
    from native_core.sparse_decode.remat_cache import reconstruct_blocks
    N, S, R, H, D = 2, 10, 1, 2, 8
    aK, aV = torch.randn(N, H, D, device=DEV), torch.randn(N, H, D, device=DEV)
    kd, vd = torch.randn(N, S, H, D, device=DEV), torch.randn(N, S, H, D, device=DEV)
    K, V = reconstruct_blocks(torch.randn(N, S, R, device=DEV), torch.zeros(N, R, H, D, device=DEV),
                              torch.randn(N, R, H, D, device=DEV), aK, aV,
                              torch.ones(N, device=DEV), torch.ones(N, device=DEV), R,
                              k_delta=kd, v_delta=vd)
    assert torch.allclose(V[:, 1:].float(), (vd + aV.unsqueeze(1)).float(), atol=1e-5)
    assert torch.allclose(V[:, 0].float(), aV.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused attention on CUDA")
@pytest.mark.parametrize("bits", [2, 4])
def test_sdpa_tile_value_codes(bits):
    from runtime.native_block_pool import (quantize_keys_per_channel, dequantize_keys_per_channel,
                                           quantize_values_per_token, dequantize_values_per_token)
    from native_core.sparse_decode.triton_fused_decode import history_attend_sdpa_tile
    torch.manual_seed(3)
    N, S, Hk, g, D, Q = 3, 64, 2, 2, 64, 5
    H = Hk * g
    dt = torch.float16
    kd = torch.randn(N, S, Hk, D, device=DEV)
    vd = torch.randn(N, S, Hk, D, device=DEV)
    kc = quantize_keys_per_channel(kd, bits, 32)
    vc = quantize_values_per_token(vd, bits, 32)
    aK, aV = torch.randn(N, Hk, D, device=DEV), torch.randn(N, Hk, D, device=DEV)
    rep = lambda t: t.repeat_interleave(g, dim=1)               # [N, Hk, ..] -> [N, H, ..]
    e = torch.empty(N, 0, device=DEV, dtype=torch.int16)
    kw = dict(U=torch.zeros(N, S, 1, device=DEV), V_K=torch.zeros(N, 1, H, D, device=DEV),
              V_V=torch.randn(N, 1, H, D, device=DEV), anchors_K=rep(aK), anchors_V=rep(aV),
              scales=torch.ones(N, device=DEV),
              cos_sliced=torch.ones(N, 1 + S, 1, D, device=DEV),
              sin_sliced=torch.zeros(N, 1 + S, 1, D, device=DEV),
              seq_lens=torch.full((N,), S, device=DEV, dtype=torch.int32), inv_scale=D ** -0.5,
              residual_K_positions=e, residual_K_values=None,
              residual_V_positions=e, residual_V_values=None,
              kq_codes=kc[0], kq_scale=kc[1], kq_zero=kc[2],
              vq_codes=vc[0], vq_scale=vc[1], vq_zero=vc[2])
    q = torch.randn(1, H, Q, D, device=DEV, dtype=dt)
    out = history_attend_sdpa_tile(q, kw, g, bits, 32, bits, 32)
    K = torch.cat([aK.unsqueeze(1), aK.unsqueeze(1)
                   + dequantize_keys_per_channel(*kc, bits, 32, torch.float32)], 1)
    V = torch.cat([aV.unsqueeze(1), aV.unsqueeze(1)
                   + dequantize_values_per_token(*vc, bits, 32, torch.float32)], 1)
    K = K.reshape(N * (1 + S), Hk, D).transpose(0, 1).repeat_interleave(g, 0)   # [H, M, D]
    V = V.reshape(N * (1 + S), Hk, D).transpose(0, 1).repeat_interleave(g, 0)
    w = torch.softmax(q[0].float() @ K.transpose(1, 2) * D ** -0.5, -1)
    ref = w @ V
    assert torch.allclose(out[0, 0].float(), ref, atol=2e-2), (out[0, 0].float() - ref).abs().max()

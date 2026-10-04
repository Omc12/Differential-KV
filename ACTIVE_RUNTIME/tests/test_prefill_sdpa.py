"""E1 (DKV_PREFILL_SDPA=1): history attention through a fused attention kernel.

history_attend_sdpa_tile must equal _prefill_fused_history_attend (the default
path) on the same inputs: GQA head-repeated tensors, real RoPE, ragged block
lengths, K and V residuals with empty slots, both residual semantics, tiled
and untiled, and the hybrid store's quantized keys.
"""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="fused attention kernel is CUDA-only")
DEV = "cuda"
DT = torch.float16


def _inputs(seed=0, N=5, S=40, Hk=2, g=4, D=32, R=6, Q=7, P=5):
    torch.manual_seed(seed)
    H = Hk * g
    rep = lambda x, dim: x.repeat_interleave(g, dim=dim)    # repeat_kv layout
    U = torch.randn(N, S, R, device=DEV) * 0.5
    V_K = rep(torch.randn(N, R, Hk, D, device=DEV), 2)
    V_V = rep(torch.randn(N, R, Hk, D, device=DEV), 2)
    aK = rep(torch.randn(N, Hk, D, device=DEV), 1)
    aV = rep(torch.randn(N, Hk, D, device=DEV), 1)
    pos = torch.arange(N * (1 + S), device=DEV).float().view(N, 1 + S) * 3.0
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=DEV).float() / D))
    ang = pos.unsqueeze(-1) * inv
    ang = torch.cat([ang, ang], -1)
    rp = torch.stack([torch.randperm(S, device=DEV)[:P] for _ in range(N)]).to(torch.int16)
    rp[:, -2:] = -1                                          # empty slots
    kw = dict(
        U=U.to(DT), V_K=V_K.to(DT), V_V=V_V.to(DT), anchors_K=aK.to(DT),
        anchors_V=aV.to(DT), scales=torch.rand(N, device=DEV) + 0.5,
        cos_sliced=ang.cos().unsqueeze(2).to(DT), sin_sliced=ang.sin().unsqueeze(2).to(DT),
        seq_lens=torch.tensor([S, S - 3, 17, S, 1][:N], device=DEV, dtype=torch.int32),
        inv_scale=D ** -0.5,
        residual_K_positions=rp, residual_K_values=rep(torch.randn(N, P, Hk, D, device=DEV), 2).to(DT),
        residual_V_positions=rp.flip(1).contiguous(),
        residual_V_values=rep(torch.randn(N, P, Hk, D, device=DEV), 2).to(DT),
    )
    q = torch.randn(1, H, Q, D, device=DEV, dtype=DT)
    return q, kw, g


def _ref(q, kw, exact):
    from native_core.sparse_decode.triton_fused_decode import (
        _prefill_fused_history_attend_compiled as f)
    return f(q=q, exact_residual=exact, lowmem=True, **kw)


def _close(a, b):
    # fp16 rounding: both paths sit ~1e-3 relative from an fp32 reference
    # (measured 0.011 on outputs of 8.8 for EACH), so the bound is relative.
    scale = max(1.0, b[0].float().abs().max().item())
    out_err = (a[0].float() - b[0].float()).abs().max().item()
    lse_err = (a[1].float() - b[1].float()).abs().max().item()
    assert out_err < 2e-3 * scale and lse_err < 2e-2, (out_err, lse_err, scale)


@pytest.mark.parametrize("exact", [False, True])
def test_matches_fused_path(exact):
    from native_core.sparse_decode.triton_fused_decode import history_attend_sdpa_tile
    q, kw, g = _inputs()
    kw["exact_residual"] = exact
    got = history_attend_sdpa_tile(q, kw, g)
    ref = _ref(q, {k: v for k, v in kw.items() if k != "exact_residual"}, exact)
    _close(got, ref)


def test_tiled_matches_untiled():
    from native_core.sparse_decode.triton_fused_decode import (
        history_attend_block_tiled, history_attend_sdpa_tile)
    q, kw, g = _inputs(seed=3)
    ref = _ref(q, kw, False)
    got = history_attend_block_tiled(lambda k: history_attend_sdpa_tile(q, k, g), kw, 2, DT)
    _close(got, ref)


def test_no_gqa():
    from native_core.sparse_decode.triton_fused_decode import history_attend_sdpa_tile
    q, kw, g = _inputs(seed=4, Hk=4, g=1)
    _close(history_attend_sdpa_tile(q, kw, g), _ref(q, kw, False))


def test_hybrid_key_codes():
    """Quantized keys: the codes path equals the fused path fed the same
    dequantized deltas."""
    from native_core.sparse_decode.triton_fused_decode import history_attend_sdpa_tile
    from runtime.native_block_pool import (
        quantize_keys_per_channel, dequantize_keys_per_channel)
    q, kw, g = _inputs(seed=5)
    N, S = kw["U"].shape[:2]
    Hk, D = kw["V_K"].shape[2] // g, kw["V_K"].shape[3]
    kd = torch.randn(N, S, Hk, D, device=DEV, dtype=DT)
    codes, sc, zp = quantize_keys_per_channel(kd, 4, 32)
    deq = dequantize_keys_per_channel(codes, sc, zp, 4, 32, DT)
    kw_ref = dict(kw, key_deltas=deq.repeat_interleave(g, dim=2))
    ref = _ref(q, kw_ref, True)
    got = history_attend_sdpa_tile(
        q, dict(kw, kq_codes=codes, kq_scale=sc, kq_zero=zp, exact_residual=True), g, 4, 32)
    _close(got, ref)

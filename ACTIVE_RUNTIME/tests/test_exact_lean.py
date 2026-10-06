"""DKV_EXACT_LEAN: exact-prefill history attention, tiled.

_dense_history_attend_tiled must equal the default path's arithmetic (all
history concatenated, RoPE, repeat_kv to every query head, one efficient
attention) on raw blocks: GQA, ragged active lengths, an anchor-only block,
several tiles, with and without the history rotation.
"""
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
DT = torch.float16


def _blocks(Hk, D, lens, seed=0):
    torch.manual_seed(seed)
    out, anchor = [], 0
    for n in lens:
        b = types.SimpleNamespace(
            anchor_idx=anchor,
            anchor_kv=torch.randn(1, 2, Hk, D, device="cuda", dtype=DT),
            active_k=(torch.randn(1, Hk, n, D, device="cuda", dtype=DT) if n else None),
            active_v=(torch.randn(1, Hk, n, D, device="cuda", dtype=DT) if n else None),
            active_k_cpu=None, active_v_cpu=None)
        out.append(b)
        anchor += 1 + n
    return out, anchor


def _tables(P, D):
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    ang = torch.arange(P, device="cuda").float()[:, None] * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos()[None].to(DT), ang.sin()[None].to(DT)       # [1, P, D]


def _reference(q, blocks, cos, sin, g, scale):
    from runtime.dkv_attention import _rope_history_k, repeat_kv
    ks, vs, pos = [], [], []
    for b in blocks:
        ks.append(b.anchor_kv[0, 0].unsqueeze(1))
        vs.append(b.anchor_kv[0, 1].unsqueeze(1))
        n = 0
        if b.active_k is not None:
            ks.append(b.active_k[0]); vs.append(b.active_v[0]); n = b.active_k.shape[2]
        pos.extend(range(b.anchor_idx, b.anchor_idx + 1 + n))
    k = torch.cat(ks, 1).unsqueeze(0)
    v = torch.cat(vs, 1).unsqueeze(0)
    pt = torch.tensor(pos, device="cuda")
    k = _rope_history_k(k, cos[0, pt][None, None], sin[0, pt][None, None])
    o, l = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, repeat_kv(k, g), repeat_kv(v, g), None, True, scale=scale)[:2]
    return o, l[..., :q.shape[2]]


@pytest.mark.parametrize("rotated_pool", ["0", "1"])
@pytest.mark.parametrize("tile", [64, 100000])
def test_tiled_equals_default(rotated_pool, tile, monkeypatch):
    monkeypatch.setenv("DKV_ROTATED_POOL", rotated_pool)
    from runtime.dkv_attention import _dense_history_attend_tiled
    Hk, g, D, Q = 2, 4, 32, 9
    blocks, P = _blocks(Hk, D, [40, 0, 63, 17, 50])
    cos, sin = _tables(P + Q, D)
    q = torch.randn(1, Hk * g, Q, D, device="cuda", dtype=DT)
    scale = D ** -0.5
    ref_o, ref_l = _reference(q, blocks, cos, sin, g, scale)
    got_o, got_l = _dense_history_attend_tiled(q, blocks, cos, sin, g, scale, tile_tokens=tile)
    sc = max(1.0, ref_o.float().abs().max().item())
    assert (got_o.float() - ref_o.float()).abs().max().item() < 2e-3 * sc
    assert (got_l.float() - ref_l.float()).abs().max().item() < 2e-2


def test_empty_history():
    from runtime.dkv_attention import _dense_history_attend_tiled
    q = torch.randn(1, 4, 3, 16, device="cuda", dtype=DT)
    assert _dense_history_attend_tiled(q, [], None, None, 2, 0.25) == (None, None)

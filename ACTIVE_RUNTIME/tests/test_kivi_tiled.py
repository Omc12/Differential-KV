"""E3: tiled KIVI-4 baseline (benchmarks/kv_baselines.py, kivi4_tiled).

_kivi_tiled_attention must equal plain attention over [dequantized history +
16-bit window] with the same causal mask: GQA, multi-tile history, prefill
chunk (Q > 1, mask given and mask None) and decode (Q = 1).
"""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
DT = torch.float16


def _setup(n_hist, n_res, Q, Hk=2, g=3, D=32, seed=0):
    import kv_baselines as KB
    torch.manual_seed(seed)
    st = {"k": [], "v": [], "rk": None, "rv": None, "n": 0}
    # history in uneven entries, as prefill chunks + decode groups produce
    left = n_hist
    while left:
        n = min(left, 96 if len(st["k"]) % 2 else 160)
        k = torch.randn(1, Hk, n, D, device="cuda", dtype=DT)
        v = torch.randn(1, Hk, n, D, device="cuda", dtype=DT)
        KB._kivi_quantize_into(st, k, v, 32, 0, 0)
        left -= n
    H = Hk * g
    q = torch.randn(1, H, Q, D, device="cuda", dtype=DT)
    kr = torch.randn(1, Hk, n_res, D, device="cuda", dtype=DT)
    vr = torch.randn(1, Hk, n_res, D, device="cuda", dtype=DT)
    lyr = types.SimpleNamespace()
    lyr.__dict__["_kivi"] = st
    KB._KIVI_TILED_CACHE["cache"] = types.SimpleNamespace(layers=[lyr])
    mod = types.SimpleNamespace(layer_idx=0)
    kh = torch.cat([(KB._unpack4(p).to(DT) * s + z).flatten(2, 3) for p, s, z in st["k"]] + [kr], -2)
    vh = torch.cat([KB._unpack4(p).to(DT) * s + z for p, s, z in st["v"]] + [vr], -2)
    return KB, mod, q, kr, vr, kh, vh, g


def _ref(q, kh, vh, g, Q):
    k = kh.repeat_interleave(g, 1).float()
    v = vh.repeat_interleave(g, 1).float()
    n = k.shape[-2]
    s = q.float() @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    s = s + torch.full((Q, n), float("-inf"), device="cuda").triu(1 + n - Q)
    return (torch.softmax(s, -1) @ v).transpose(1, 2)


@pytest.mark.parametrize("n_hist,n_res,Q,use_mask", [
    (800, 200, 64, True),     # prefill chunk, mask given
    (0, 64, 64, False),       # first chunk: no history, mask None
    (800, 130, 1, False),     # decode
])
def test_tiled_matches_reference(n_hist, n_res, Q, use_mask, monkeypatch):
    KB, mod, q, kr, vr, kh, vh, g = _setup(n_hist, n_res, Q)
    monkeypatch.setattr(KB, "KIVI_TILE_TOKENS", 200)          # force several tiles
    mask = None
    if use_mask:
        n = kh.shape[-2]
        mask = torch.ones(Q, n, dtype=torch.bool, device="cuda").tril(n - Q)[None, None]
    got, _ = KB._kivi_tiled_attention(mod, q, kr, vr, mask, scaling=q.shape[-1] ** -0.5)
    ref = _ref(q, kh, vh, g, Q)
    assert (got.float() - ref).abs().max().item() < 5e-3

"""dense_tiled baseline (benchmarks/kv_baselines.py): a 16-bit cache held in
chunks and read by the tiled KIVI attention. It must equal plain attention over
the same keys and values (prefill chunk with a causal mask, and decode), and
store exactly the 16-bit bytes.
"""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
DT = torch.float16


def _ref(q, k, v, Q, g, scale):
    kk = k.repeat_interleave(g, 1).float()
    vv = v.repeat_interleave(g, 1).float()
    s = q.float() @ kk.transpose(-1, -2) * scale
    n = k.shape[-2]
    if Q > 1:
        s = s + torch.full((Q, n), float("-inf"), device=s.device).triu(1 + n - Q)
    return (torch.softmax(s, -1) @ vv).transpose(1, 2)


@pytest.mark.parametrize("Q", [1, 64])
def test_dense_tiled_equals_plain_attention(Q):
    import kv_baselines as KB
    torch.manual_seed(0)
    Hk, g, D = 2, 3, 32
    H = Hk * g
    st = {"k": [], "v": [], "rk": None, "rv": None, "n": 0}
    hist_k, hist_v = [], []
    for n in (160, 96, 9000, 64):                      # more than one 8k tile
        k = torch.randn(1, Hk, n, D, device="cuda", dtype=DT)
        v = torch.randn(1, Hk, n, D, device="cuda", dtype=DT)
        rk, rv = KB._kivi_quantize_into(st, k, v, 32, 0, 0, bits=16)
        hist_k.append(k[..., :k.shape[-2] - rk.shape[-2], :])
        hist_v.append(v[..., :v.shape[-2] - rv.shape[-2], :])
        hist_k.append(rk)
        hist_v.append(rv)
    # keep history = everything stored; the leftover (<32) rows go to the window
    stored_k = torch.cat([p for p, _, _ in st["k"]], -2)
    stored_v = torch.cat([p for p, _, _ in st["v"]], -2)
    kr = torch.randn(1, Hk, Q + 16, D, device="cuda", dtype=DT)
    vr = torch.randn(1, Hk, Q + 16, D, device="cuda", dtype=DT)
    lyr = types.SimpleNamespace(kivi_bits=16)
    lyr.__dict__["_kivi"] = st
    KB._KIVI_TILED_CACHE["cache"] = types.SimpleNamespace(layers=[lyr])
    q = torch.randn(1, H, Q, D, device="cuda", dtype=DT)
    out, _ = KB._kivi_tiled_attention(types.SimpleNamespace(layer_idx=0), q, kr, vr, None)
    ref = _ref(q, torch.cat([stored_k, kr], -2), torch.cat([stored_v, vr], -2), Q, g,
               D ** -0.5)
    assert torch.allclose(out.float(), ref, atol=2e-3), (out.float() - ref).abs().max()


def test_dense_tiled_bytes_are_16_bit():
    import kv_baselines as KB
    st = {"k": [], "v": [], "rk": None, "rv": None, "n": 0}
    k = torch.randn(1, 2, 1000, 32, device="cuda", dtype=DT)
    rk, rv = KB._kivi_quantize_into(st, k, k.clone(), 32, 128, 0, bits=16)
    lyr = types.SimpleNamespace()
    lyr.__dict__["_kivi"] = dict(st, rk=rk, rv=rv)
    nbytes = KB._KiviLayerMixin.kivi_bytes(lyr)
    assert nbytes == 2 * 1000 * 2 * 32 * 2                # K and V, all 16-bit

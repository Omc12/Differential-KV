"""NoPE layers (use_rope=False: SmolLM3, Llama 4) must not be rotated by DKV.

V13 panel: SmolLM3-3B at 32k agreed with dense on 0/5 first tokens (KL 9.2)
because DKV rotated q/k and stored history on its 9 NoPE layers. The fix swaps
every rotary table a NoPE layer's forward builds for the identity.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime import dkv_attention as da  # noqa: E402


def test_layer_uses_rope_reads_the_module_flag():
    assert da.layer_uses_rope(types.SimpleNamespace(use_rope=False)) is False
    assert da.layer_uses_rope(types.SimpleNamespace(use_rope=True)) is True
    # SmolLM3 stores config.no_rope_layers[i]: the INT 0 / 1, not a bool.
    assert da.layer_uses_rope(types.SimpleNamespace(use_rope=0)) is False
    assert da.layer_uses_rope(types.SimpleNamespace(use_rope=1)) is True
    # Every other architecture has no flag: rotate as before.
    assert da.layer_uses_rope(types.SimpleNamespace()) is True


def test_identity_tables_make_every_rotation_a_no_op():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 7, 64)
    k = torch.randn(1, 2, 7, 64)
    pos = torch.arange(7).unsqueeze(0)
    inv = 1.0 / (10000 ** (torch.arange(0, 48, 2).float() / 48))
    ang = pos[..., None].float() * inv                      # partial rotary: 48 of 64
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos(), emb.sin()
    qr, kr = da.apply_rotary_pos_emb(q, k, cos, sin)
    assert not torch.allclose(kr, k)                        # real tables rotate
    c0, s0 = da._nope_tables(cos, sin)
    assert c0.shape == cos.shape and s0.shape == sin.shape
    qn, kn = da.apply_rotary_pos_emb(q, k, c0, s0)
    assert torch.equal(qn, q) and torch.equal(kn, k)
    assert torch.equal(da._apply_rope_single(k, c0.unsqueeze(1), s0.unsqueeze(1)), k)


def test_identity_tables_are_cached_and_grown_not_rebuilt():
    da._NOPE_TABLES.clear()
    cos = torch.randn(1, 10, 32)
    a, _ = da._nope_tables(cos, cos)
    b, _ = da._nope_tables(cos[:, :4], cos[:, :4])
    assert b.shape[-2] == 4 and b.data_ptr() == a.data_ptr()   # a view, no new table
    c, _ = da._nope_tables(torch.randn(1, 20, 32), torch.randn(1, 20, 32))
    assert c.shape[-2] == 20
    assert da._nope_tables(None, None) == (None, None)


def test_resid_attn_weight_does_not_rotate_nope_layer_keys(monkeypatch):
    """Residual-row ranking re-rotates stored keys outside the forward; it reads
    the manager's NoPE registry."""
    calls = {}
    real = da.apply_rotary_pos_emb

    def spy(q, k, cos, sin, *a, **kw):
        calls["sin_abs"] = float(sin.abs().sum())
        return real(q, k, cos, sin, *a, **kw)

    monkeypatch.setattr(da, "apply_rotary_pos_emb", spy)
    D, T = 8, 3

    def rotary(x, p):
        e = p[..., None].float().expand(*p.shape, D) * 0.5
        return e.cos(), e.sin()

    Q = torch.randn(2, 2, D)
    P = torch.tensor([5, 6])
    mgr = types.SimpleNamespace(_qres={"s": {3: (Q, P, None, 1.0, None)}},
                                _qres_rotary=rotary, _nope_layers={3})
    blk = types.SimpleNamespace(session_id="s", layer_idx=3, token_indices=[0, 1, 2])
    da.resid_attn_weight(mgr, blk, torch.randn(T, 2 * D))
    assert calls.get("sin_abs") == 0.0
    mgr._nope_layers = set()
    da.resid_attn_weight(mgr, blk, torch.randn(T, 2 * D))
    assert calls["sin_abs"] > 0.0

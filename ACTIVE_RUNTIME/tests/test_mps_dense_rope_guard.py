"""The MPS dense window must not be rotated when the pool already stores rotated K.

`_is_mps_decode` hands the dense window to the Metal shader together with
cos/sin, and the shader rotates it. But `dense_k_assembled` is built from the
blocks' `active_k`, so under DKV_ROTATED_POOL=1 those rows are ALREADY post-RoPE
and the shader's rotation is a SECOND one.

Reachable in production rather than hypothetically: `low` is the only preset that
keeps rotated_pool=True (native_core/config.py:143) and it also sets
approximate_attn=True on macOS (config.py:37) -- which is precisely the
_is_mps_decode gate. CUDA cannot reach any of this; the branch requires MPS.

WHY THIS NEEDS A TEST AT ALL: RoPE is orthogonal. A double rotation preserves
every norm exactly, so nothing raises, nothing looks out of range, and the only
damage is to ANGLES. Asserting on norms would pass while the bug is present --
these tests compare against the singly-rotated reference instead, and the
source-level test pins the guard itself so the two call sites cannot drift apart
again.
"""
import os
import re

import pytest
import torch

from native_core.sparse_decode.triton_fused_decode import _partial_rope_apply

HERE = os.path.dirname(os.path.abspath(__file__))
ATTN = os.path.join(os.path.dirname(HERE), "runtime", "dkv_attention.py")


def _rope_tables(seq, rotary_dim, device="cpu", dtype=torch.float32):
    pos = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device,
                                          dtype=torch.float32) / rotary_dim))
    ang = pos * inv.unsqueeze(0)
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def test_double_rotation_preserves_norms_so_norms_cannot_detect_it():
    """Guards the guard: proves a norm-based assertion would MISS this bug."""
    torch.manual_seed(0)
    k = torch.randn(1, 2, 8, 64)
    cos, sin = _rope_tables(8, 32)
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)

    once = _partial_rope_apply(k, c, s)
    twice = _partial_rope_apply(once, c, s)

    # Norms are identical to float precision -- RoPE is orthogonal.
    torch.testing.assert_close(once.norm(dim=-1), twice.norm(dim=-1),
                               rtol=1e-5, atol=1e-5)
    # But the vectors themselves are NOT. That difference is the whole bug.
    assert not torch.allclose(once, twice, rtol=1e-3, atol=1e-3), (
        "double rotation produced identical keys -- the rope tables are "
        "degenerate and this test proves nothing")


def test_second_rotation_is_a_real_angle_error():
    """The corruption is large enough to matter for attention scores."""
    torch.manual_seed(1)
    k = torch.randn(1, 2, 16, 64)
    q = torch.randn(1, 2, 1, 64)
    cos, sin = _rope_tables(16, 32)
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)

    once = _partial_rope_apply(k, c, s)
    twice = _partial_rope_apply(once, c, s)

    good = (q @ once.transpose(-1, -2)).squeeze()
    bad = (q @ twice.transpose(-1, -2)).squeeze()
    # argmax over positions moves, i.e. routing/attention actually changes.
    assert (good - bad).abs().max().item() > 1e-2, (
        "a second rotation left scores unchanged -- partial-RoPE coverage may "
        "be zero here, so this test would not catch a regression")


@pytest.mark.parametrize("marker", [
    "if dense_k_assembled is not None and not _pool_rotated_k():",
    "dense_k_rot = (dense_k_valid if _pool_rotated_k()",
])
def test_both_mps_sites_keep_the_guard(marker):
    """Source-level pin.

    Neither site is reachable from CPU or CUDA, so a behavioural test cannot
    cover them on the machines that usually run this suite. Pin the guard text
    instead: the production kernel path and the _validate_this_step path must
    BOTH consult _pool_rotated_k(), or the validator disagrees with production
    for a reason of its own making.
    """
    src = open(ATTN).read()
    assert marker in src, (
        f"the MPS double-rotation guard is missing: {marker!r} not found in "
        f"dkv_attention.py. A dense window rotated twice preserves norms and "
        f"only corrupts angles, so nothing else in this suite will notice.")


def test_metal_binding_can_attend_dense_without_rotating_it():
    """The guard disables rotation by passing EMPTY cos/sin, which only works
    because metal_runtime.mm derives has_dense_rope separately from has_dense.
    If those were ever collapsed into one flag, the guard would silently drop
    the dense window instead of merely leaving it unrotated."""
    mm = os.path.join(os.path.dirname(HERE), "native_core", "dkv_core", "src",
                      "metal_runtime.mm")
    src = open(mm).read()
    m = re.search(r"bool\s+has_dense_rope\s*=\s*([^;]+);", src)
    assert m, "has_dense_rope not found in metal_runtime.mm"
    expr = " ".join(m.group(1).split())
    assert "cos_dense" in expr and "numel() > 0" in expr, (
        f"has_dense_rope no longer keys off cos_dense being non-empty "
        f"(found: {expr!r}) -- passing empty cos/sin may now skip the dense "
        f"window entirely rather than just its rotation.")

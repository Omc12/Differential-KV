"""torch's efficient SDPA kernel is wrong on stride-0 (expanded) K/V with a mask.

V13 panel: every gemma-4 model showed first-token KL 0.9-2.6. Per-layer diff
traced it to transformers' NATIVE sliding layers (one KV head, so repeat_kv
returns a stride-0 view) through torch's memory-efficient kernel at DKV's
1025-token prefill chunk. The guard makes such K/V contiguous.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime import dkv_attention as da  # noqa: E402


def _inputs(qlen=1025, D=256, H=8, W=512, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(1, qlen, H, D, device=device, dtype=torch.bfloat16, generator=g).transpose(1, 2)
    k1 = torch.randn(1, 1, qlen, D, device=device, dtype=torch.bfloat16, generator=g) * 0.3
    v1 = torch.randn(1, 1, qlen, D, device=device, dtype=torch.bfloat16, generator=g)
    k, v = k1.expand(1, H, qlen, D), v1.expand(1, H, qlen, D)       # stride 0 across heads
    pos = torch.arange(qlen, device=device)
    m = ((pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < W))[None, None]
    return q, k, v, m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel defect")
def test_guard_makes_expanded_kv_masked_sdpa_match_math():
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v, m = _inputs()
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m, scale=1.0)
    da.install_sdpa_expanded_kv_guard()
    assert getattr(F.scaled_dot_product_attention, "_dkv_expanded_kv_guard", False)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=1.0)
    rel = float((out.float() - ref).norm() / ref.norm())
    assert rel < 5e-3, rel
    # Without the guard (the wrapped original) this input is the defect.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        raw = F.scaled_dot_product_attention.__wrapped__(q, k, v, attn_mask=m, scale=1.0)
    raw_rel = float((raw.float() - ref).norm() / ref.norm())
    print(f"guarded {rel:.1e}  unguarded {raw_rel:.1e}")


def test_guard_is_idempotent_and_leaves_other_calls_alone():
    import torch.nn.functional as F
    da.install_sdpa_expanded_kv_guard()
    first = F.scaled_dot_product_attention
    assert da.install_sdpa_expanded_kv_guard() is False
    assert F.scaled_dot_product_attention is first
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    ref = first.__wrapped__(q, k, v, is_causal=True)
    assert torch.equal(F.scaled_dot_product_attention(q, k, v, is_causal=True), ref)

"""Prefill block router (dkv_attention._sparse_prefill_filter_blocks) vs MLX.

CUDA runs block-sparse attention DURING prefill, so a block this router drops is
a block the model never reads -- its hidden states, and therefore every later
query, are those of a model that never saw those tokens. That makes the router's
SCORING, not its K, the correctness surface, and it is what these tests pin.

The reference is mlx_dkv_wrapper._block_relevance_minmax (:1098), which is what
MLX's _sparse_prefill_attend (:1265) actually calls -- NOT the residual router it
uses at decode. Written out literally, per head and per chunk token:

    rel(block) = max over (h, l) of  sum_d max(q[h,l,d]*min[b,h_kv,d],
                                               q[h,l,d]*max[b,h_kv,d]) * scale

test_matches_mlx_literal_formula is that expression transcribed directly from
the MLX source and compared against the shipped implementation, which computes
the same quantity as two GEMMs.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from runtime.dkv_attention import (                          # noqa: E402
    _prefill_block_key_boxes,
    _sparse_prefill_filter_blocks,
    _sparse_prefill_relevance,
)


class _FakeBlock:
    """Minimal stand-in for StreamingKVBlock's routing surface."""

    def __init__(self, anchor_idx, anchor_k, active_k, state="ACCUMULATING"):
        H_kv, D = anchor_k.shape
        self.anchor_idx = anchor_idx
        # StreamingKVBlock.anchor_kv is [1, 2, H_kv, D] (K and V stacked).
        self.anchor_kv = torch.stack([anchor_k, torch.zeros_like(anchor_k)],
                                     dim=0).unsqueeze(0)
        self.active_k = None if active_k is None else active_k.unsqueeze(0)
        self.active_k_cpu = None
        self.state = state


def _mlx_literal_minmax(q, k_min, k_max, scale):
    """mlx_dkv_wrapper._block_relevance_minmax, transcribed elementwise.

    q [H_q, L, D]; k_min/k_max [nb, H_kv, D]; returns [nb].
    """
    H_q, L, D = q.shape
    nb, H_kv, _ = k_min.shape
    gpk = max(1, H_q // H_kv)
    # MIN_exp/MAX_exp: [H_kv, 1, 1, nb, D];  q_exp: [H_kv, gpk, L, 1, D]
    MIN_exp = k_min.permute(1, 0, 2)[:, None, None, :, :].float()
    MAX_exp = k_max.permute(1, 0, 2)[:, None, None, :, :].float()
    q_exp = q.reshape(H_kv, gpk, L, D)[:, :, :, None, :].float()
    bound = torch.maximum(q_exp * MIN_exp, q_exp * MAX_exp).sum(-1) * scale
    return bound.reshape(H_q, L, nb).amax(dim=(0, 1))


def _make_blocks(nb, H_kv=2, D=16, T=8, gen=None):
    g = gen or torch.Generator().manual_seed(0)
    blocks = []
    for i in range(nb):
        anchor_k = torch.randn(H_kv, D, generator=g)
        active_k = torch.randn(H_kv, T, D, generator=g)
        blocks.append(_FakeBlock(i * (T + 1), anchor_k, active_k))
    return blocks


def test_matches_mlx_literal_formula():
    """The two-GEMM form is an identity, not an approximation.

    max(a, b) == (a + b)/2 + |a - b|/2, so with a = q_d*min_d, b = q_d*max_d and
    max_d >= min_d the elementwise sum collapses to q.mid + |q|.half.
    """
    g = torch.Generator().manual_seed(7)
    H_q, H_kv, D, L, nb = 8, 2, 16, 5, 6
    blocks = _make_blocks(nb, H_kv=H_kv, D=D, gen=g)
    k_min, k_max = _prefill_block_key_boxes(blocks, torch.device("cpu"))
    chunk_q = torch.randn(1, H_q, L, D, generator=g)
    scale = 1.0 / math.sqrt(D)

    got = _sparse_prefill_relevance(chunk_q, k_min, k_max, scale)
    want = _mlx_literal_minmax(chunk_q[0], k_min, k_max, scale)

    assert got.shape == (nb,)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_token_tiling_does_not_change_the_score():
    """Tiling the chunk-token axis is a memory bound, not a semantic one."""
    g = torch.Generator().manual_seed(11)
    blocks = _make_blocks(5, gen=g)
    k_min, k_max = _prefill_block_key_boxes(blocks, torch.device("cpu"))
    chunk_q = torch.randn(1, 4, 37, 16, generator=g)
    untiled = _sparse_prefill_relevance(chunk_q, k_min, k_max, 0.25, tok_tile=64)
    tiled = _sparse_prefill_relevance(chunk_q, k_min, k_max, 0.25, tok_tile=8)
    torch.testing.assert_close(untiled, tiled, rtol=1e-6, atol=1e-6)


def test_key_box_is_the_true_min_max_over_anchor_and_active_keys():
    g = torch.Generator().manual_seed(3)
    H_kv, D, T = 2, 16, 8
    anchor_k = torch.randn(H_kv, D, generator=g)
    active_k = torch.randn(H_kv, T, D, generator=g)
    b = _FakeBlock(0, anchor_k, active_k)

    k_min, k_max = _prefill_block_key_boxes([b], torch.device("cpu"))
    all_keys = torch.cat([anchor_k.unsqueeze(1), active_k], dim=1)  # [H_kv, 1+T, D]

    torch.testing.assert_close(k_min[0], all_keys.amin(dim=1))
    torch.testing.assert_close(k_max[0], all_keys.amax(dim=1))


def test_key_box_cache_invalidates_when_a_block_grows():
    """Blocks accumulate during prefill; a stale box would rank on old content."""
    g = torch.Generator().manual_seed(5)
    H_kv, D = 2, 16
    anchor_k = torch.zeros(H_kv, D)
    b = _FakeBlock(0, anchor_k, torch.zeros(H_kv, 4, D))
    _prefill_block_key_boxes([b], torch.device("cpu"))
    assert getattr(b, "_sp_key_box", None) is not None

    grown = torch.zeros(H_kv, 5, D)
    grown[:, 4, :] = 9.0
    b.active_k = grown.unsqueeze(0)
    k_min, k_max = _prefill_block_key_boxes([b], torch.device("cpu"))
    assert float(k_max[0].max()) == pytest.approx(9.0)


def test_needle_buried_in_a_block_survives_routing():
    """The regression the whole change exists for.

    One block holds a key that matches the query strongly, at an interior offset
    -- a passcode buried mid-block. Its ANCHOR is as generic as every other
    block's, which is exactly the case the previous router
    (`anchor_ks . chunk_q.mean(dim=(0,1))`) could not see: it scored the block by
    its first token and ranked the needle's block by its prose.
    """
    g = torch.Generator().manual_seed(13)
    H_q, H_kv, D, T = 4, 2, 32, 16
    nb = 24
    needle_block = 17
    needle_off = 13                       # deep inside the block, not the anchor

    blocks = []
    for i in range(nb):
        anchor_k = torch.randn(H_kv, D, generator=g) * 0.1
        active_k = torch.randn(H_kv, T, D, generator=g) * 0.1
        blocks.append(_FakeBlock(i * (T + 1), anchor_k, active_k))

    # The query direction, planted only on one interior token of one block.
    q_dir = torch.randn(D, generator=g)
    q_dir = q_dir / q_dir.norm()
    blocks[needle_block].active_k[0, :, needle_off, :] = q_dir * 8.0

    chunk_q = torch.randn(1, H_q, 12, D, generator=g) * 0.1
    chunk_q[0, 1, 5, :] = q_dir * 4.0     # one retrieval head, one token

    k_min, k_max = _prefill_block_key_boxes(blocks, torch.device("cpu"))
    rel = _sparse_prefill_relevance(chunk_q, k_min, k_max, 1.0 / math.sqrt(D))
    assert int(rel.argmax()) == needle_block, "needle block must rank first"

    # And the old scoring must NOT find it -- otherwise this test proves nothing
    # about the fix (a test that passes both before and after is not a test).
    anchor_ks = torch.stack([b.anchor_kv[0, 0] for b in blocks], dim=0).float()
    q_repr = chunk_q[0].mean(dim=(0, 1)).float()
    old = torch.einsum("nhd,d->nh", anchor_ks, q_repr).mean(dim=1)
    assert int(old.argmax()) != needle_block, "old router would have ranked it first"


def test_end_to_end_filter_keeps_the_needle_block():
    """Same needle, through the real entry point with sinks + recency window."""
    g = torch.Generator().manual_seed(17)
    H_q, H_kv, D, T = 4, 2, 32, 16
    nb, needle_block, needle_off = 64, 20, 11
    stride = T + 1

    blocks = []
    for i in range(nb):
        anchor_k = torch.randn(H_kv, D, generator=g) * 0.1
        active_k = torch.randn(H_kv, T, D, generator=g) * 0.1
        blocks.append(_FakeBlock(i * stride, anchor_k, active_k))

    q_dir = torch.randn(D, generator=g)
    q_dir = q_dir / q_dir.norm()
    blocks[needle_block].active_k[0, :, needle_off, :] = q_dir * 8.0
    chunk_q = torch.randn(1, H_q, 12, D, generator=g) * 0.1
    chunk_q[0, 1, 5, :] = q_dir * 4.0

    env = {"DKV_SPARSE_PREFILL_MIN": "0", "DKV_SPARSE_PREFILL_WINDOW": "64",
           "DKV_SPARSE_PREFILL_KMIN": "4", "DKV_SPARSE_PREFILL_FRAC": "0.1",
           "DKV_ROTATED_POOL": "1"}
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        kept = _sparse_prefill_filter_blocks(blocks, chunk_q, sink_blocks=1,
                                             chunk_start=nb * stride)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert len(kept) < len(blocks), "must actually be sparse (handoff §3 trap)"
    assert blocks[needle_block] in kept, "needle block was routed away at prefill"
    anchors = [b.anchor_idx for b in kept]
    assert anchors == sorted(anchors), "downstream builds positions from anchor_idx"


def test_unrotated_pool_falls_back_to_attending_everything():
    """With DKV_ROTATED_POOL=0 the pool holds PRE-RoPE keys while chunk_q is
    POST-RoPE. Scoring across frames is meaningless, so the router must decline
    rather than emit a ranking with no defined relationship to the query.
    """
    g = torch.Generator().manual_seed(23)
    blocks = _make_blocks(40, gen=g)
    chunk_q = torch.randn(1, 4, 8, 16, generator=g)
    old = os.environ.get("DKV_ROTATED_POOL")
    os.environ["DKV_ROTATED_POOL"] = "0"
    os.environ["DKV_SPARSE_PREFILL_MIN"] = "0"
    try:
        kept = _sparse_prefill_filter_blocks(blocks, chunk_q, sink_blocks=1,
                                             chunk_start=40 * 9)
    finally:
        os.environ.pop("DKV_SPARSE_PREFILL_MIN", None)
        if old is None:
            os.environ.pop("DKV_ROTATED_POOL", None)
        else:
            os.environ["DKV_ROTATED_POOL"] = old
    assert kept is blocks


def test_disabled_and_below_min_ctx_are_pass_through():
    g = torch.Generator().manual_seed(29)
    blocks = _make_blocks(40, gen=g)
    chunk_q = torch.randn(1, 4, 8, 16, generator=g)

    old = os.environ.get("DKV_SPARSE_PREFILL")
    os.environ["DKV_SPARSE_PREFILL"] = "0"
    try:
        assert _sparse_prefill_filter_blocks(blocks, chunk_q, 1, 99999) is blocks
    finally:
        if old is None:
            os.environ.pop("DKV_SPARSE_PREFILL", None)
        else:
            os.environ["DKV_SPARSE_PREFILL"] = old

    # Below DKV_SPARSE_PREFILL_MIN (default 2048) prefill stays fully dense.
    assert _sparse_prefill_filter_blocks(blocks, chunk_q, 1, 512) is blocks

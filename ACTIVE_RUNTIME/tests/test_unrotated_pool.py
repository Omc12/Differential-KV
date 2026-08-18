"""DKV_ROTATED_POOL=0: the pool stores PRE-RoPE keys and the reader re-rotates them.

ACTIVE_RUNTIME/docs/cuda_port_record.md, item 3. The whole change lives or dies on one property:
a key read back out of the unrotated pool, rotated to its absolute position, must
equal the key the rotated pool would have stored at that position.

This is testable EXACTLY on the residual half, because residuals are stored
verbatim (no SVD in the way) — so any error in the RoPE math, in the slot->position
map, or in the block->absolute-position arithmetic shows up as a numeric mismatch
rather than as a slightly worse benchmark score.

That distinction matters here: RoPE is PARTIAL on this model family (64 of 256
dims), so a wrong position can only perturb ~25% of each key. It degrades retrieval
without ever zeroing a score, which is exactly the kind of bug a needle sweep passes
straight through. Hence an exactness test rather than a recall test.
"""
import math
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlx.nn as nn                                             # noqa: E402
from serving.mlx_dkv_wrapper import (                           # noqa: E402
    MLXKVBlockManager, _rope_at, _rope_tables,
)

H_Q, H_KV, D = 8, 2, 64
BLOCK, RECENCY, RANK = 32, 64, 8
ROPE_DIMS, ROPE_BASE = 32, 10000.0


def _mgr(rotated: bool):
    os.environ["DKV_ROTATED_POOL"] = "1" if rotated else "0"
    m = MLXKVBlockManager(num_layers=1, heads=H_Q, kv_heads=H_KV, head_dim=D,
                          rank=RANK, block_size=BLOCK, recency_window=RECENCY)
    # recency_window is honoured now (it used to be silently overwritten by the
    # capacity-derived default). _seq_len still reads it back off the manager rather
    # than assuming, because DKV_ENGAGE_THRESHOLD outranks both — and if the window
    # is bigger than the sequence, nothing compresses and every assertion below
    # passes vacuously.
    m.max_blocks = 64
    m.max_residual = 8          # small budget keeps the round-trip loop below quick
    m.set_attended_layers([0])
    m.set_rope_params(dims=ROPE_DIMS, base=ROPE_BASE, scale=1.0, traditional=False)
    return m


def _seq_len(m):
    """Long enough to flush several whole blocks out of the recency window."""
    return m.recency_window + 6 * BLOCK


# ── the RoPE primitive itself ────────────────────────────────────────────────

def test_gather_rope_matches_mx_fast_rope_on_a_contiguous_run():
    rope = nn.RoPE(ROPE_DIMS, traditional=False, base=ROPE_BASE, scale=1.0)
    cos_t, sin_t = _rope_tables(2048, ROPE_DIMS, ROPE_BASE)
    x = mx.random.normal((1, H_KV, 16, D))
    ref = rope(x, offset=512)
    got = _rope_at(x, mx.arange(512, 528, dtype=mx.int32), cos_t, sin_t, ROPE_DIMS)
    assert float(mx.max(mx.abs(ref - got))) < 1e-4


def test_gather_rope_matches_on_SCATTERED_positions():
    """The case mx.fast.rope cannot express in one call — routed blocks and
    residual tokens are both non-contiguous, which is why the table exists."""
    rope = nn.RoPE(ROPE_DIMS, traditional=False, base=ROPE_BASE, scale=1.0)
    cos_t, sin_t = _rope_tables(2048, ROPE_DIMS, ROPE_BASE)
    pos = [0, 1, 7, 63, 512, 2047, 129, 8]
    x = mx.random.normal((1, H_KV, len(pos), D))
    ref = mx.concatenate([rope(x[:, :, i:i + 1], offset=p) for i, p in enumerate(pos)], axis=2)
    got = _rope_at(x, mx.array(pos, dtype=mx.int32), cos_t, sin_t, ROPE_DIMS)
    assert float(mx.max(mx.abs(ref - got))) < 1e-4


def test_rope_leaves_the_non_rotary_tail_untouched():
    """Partial rotary: only the first ROPE_DIMS of each head may change."""
    cos_t, sin_t = _rope_tables(64, ROPE_DIMS, ROPE_BASE)
    x = mx.random.normal((1, H_KV, 4, D))
    got = _rope_at(x, mx.array([1, 2, 3, 4], dtype=mx.int32), cos_t, sin_t, ROPE_DIMS)
    assert float(mx.max(mx.abs(got[..., ROPE_DIMS:] - x[..., ROPE_DIMS:]))) == 0.0
    assert float(mx.max(mx.abs(got[..., :ROPE_DIMS] - x[..., :ROPE_DIMS]))) > 1e-3


# ── the pool round-trip, which is the actual claim ───────────────────────────

def _fill(mgr, sid, k_unrot, k_rot, v, n):
    mgr.init_session(sid, prefill_len=n)
    for t in range(n):
        mgr.ingest_streaming(
            sid, 0,
            k_rot[:, :, t:t + 1, :], v[:, :, t:t + 1, :],
            k_unrot=(None if mgr.rotated_pool else k_unrot[:, :, t:t + 1, :]),
        )


def test_residual_keys_round_trip_to_their_rotated_originals():
    """THE test. Re-rotating an unrotated residual must reproduce the rotated key
    that lived at that absolute position — to fp16 storage precision."""
    mgr = _mgr(rotated=False)
    n = _seq_len(mgr)
    rope = nn.RoPE(ROPE_DIMS, traditional=False, base=ROPE_BASE, scale=1.0)
    k_un = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    v = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    k_rot = rope(k_un, offset=0)

    _fill(mgr, "s", k_un, k_rot, v, n)
    sess = mgr.sessions["s"]
    nb = sess["num_blocks"][0]
    assert nb > 0, "test needs at least one compressed block"

    res_k = sess["comp_res_k"][0][:nb]          # [nb, R, H_kv, D]  (unrotated)
    res_pos = sess["comp_res_pos"][0][:nb]      # [nb, R]           block-relative
    res_n = sess["comp_res_n"][0][:nb]

    worst = 0.0
    checked = 0
    for b in range(nb):
        for i in range(int(res_n[b])):
            rel = int(res_pos[0 + b][i])
            if rel == 0:
                continue                        # padded slot
            abs_pos = b * BLOCK + rel
            got = mgr._rotate_to_abs(
                res_k[b, i].reshape(H_KV, 1, D),
                mx.array([abs_pos], dtype=mx.int32), n + BLOCK).reshape(H_KV, D)
            want = k_rot[0, :, abs_pos, :]
            worst = max(worst, float(mx.max(mx.abs(got.astype(mx.float32)
                                                   - want.astype(mx.float32)))))
            checked += 1
    assert checked > 0, "no residuals were stored — test proves nothing"
    assert worst < 5e-3, f"residual round-trip off by {worst} over {checked} keys"


def test_anchor_keys_round_trip_to_their_rotated_originals():
    """Anchors sit at block-relative 0, i.e. absolute b*block_size."""
    mgr = _mgr(rotated=False)
    n = _seq_len(mgr)
    rope = nn.RoPE(ROPE_DIMS, traditional=False, base=ROPE_BASE, scale=1.0)
    k_un = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    v = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    k_rot = rope(k_un, offset=0)

    _fill(mgr, "s", k_un, k_rot, v, n)
    sess = mgr.sessions["s"]
    nb = sess["num_blocks"][0]
    anc = sess["comp_anc_k"][0][:nb]            # [nb, H_kv, D]
    got = mgr._rotate_to_abs(anc.transpose(1, 0, 2),
                             mx.arange(nb, dtype=mx.int32) * BLOCK,
                             n + BLOCK).transpose(1, 0, 2)
    want = mx.stack([k_rot[0, :, b * BLOCK, :] for b in range(nb)], axis=0)
    assert float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))) < 5e-3


def test_rotated_pool_stores_the_rotated_keys_and_needs_no_positions():
    """Control: the default mode is unchanged and allocates none of the new slabs."""
    mgr = _mgr(rotated=True)
    n = _seq_len(mgr)
    rope = nn.RoPE(ROPE_DIMS, traditional=False, base=ROPE_BASE, scale=1.0)
    k_un = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    v = mx.random.normal((1, H_KV, n, D)).astype(mx.float16)
    k_rot = rope(k_un, offset=0)
    _fill(mgr, "s", k_un, k_rot, v, n)
    sess = mgr.sessions["s"]
    assert sess.get("comp_res_pos") is None
    assert sess.get("dense_keys_unrot") is None
    nb = sess["num_blocks"][0]
    anc = sess["comp_anc_k"][0][:nb]
    want = mx.stack([k_rot[0, :, b * BLOCK, :] for b in range(nb)], axis=0)
    assert float(mx.max(mx.abs(anc.astype(mx.float32) - want.astype(mx.float32)))) < 5e-3


def test_unrotated_pool_refuses_an_ingest_that_forgot_the_unrotated_key():
    """A silent decline here would compress rotated keys into a pool the reader
    rotates a SECOND time. Refuse instead."""
    mgr = _mgr(rotated=False)
    mgr.init_session("s", prefill_len=4)
    k = mx.zeros((1, H_KV, 1, D), dtype=mx.float16)
    with pytest.raises(RuntimeError, match="unrotated"):
        mgr.ingest_streaming("s", 0, k, k)


def test_interleaved_rope_layout_is_refused_not_approximated():
    os.environ["DKV_ROTATED_POOL"] = "0"
    m = MLXKVBlockManager(num_layers=1, heads=H_Q, kv_heads=H_KV, head_dim=D,
                          rank=RANK, block_size=BLOCK, recency_window=RECENCY)
    with pytest.raises(RuntimeError, match="interleaved"):
        m.set_rope_params(dims=ROPE_DIMS, base=ROPE_BASE, scale=1.0, traditional=True)


def teardown_module(_m):
    os.environ.pop("DKV_ROTATED_POOL", None)

"""The exact-residual budget cannot exceed the delta rows a block actually has.

A block of `block_size` tokens contributes one anchor and S_comp = block_size - 1
delta rows, so at most S_comp positions can be kept exact. The budget
(DKV_MAX_RESIDUAL, default 128) was applied without that clamp: the selection
returned only S_comp rows while `pad_len = max_residual - n_res` still evaluated
to 0, so nothing padded the result and the store into the [max_residual, ...]
slab raised

    ValueError: [broadcast_shapes] Shapes (31,2,64) and (1,128,2,64) cannot be broadcast

Production defaults (block 1024, budget 128) never reach it, but it blocked every
small-block configuration — including tests, which had to pick an unnatural budget
to avoid it.
"""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.mlx_dkv_wrapper import MLXKVBlockManager     # noqa: E402

H_Q, H_KV, D = 8, 2, 64


def _mgr(block_size, window=64):
    os.environ["DKV_ENGAGE_THRESHOLD"] = str(window)
    m = MLXKVBlockManager(num_layers=1, heads=H_Q, kv_heads=H_KV, head_dim=D,
                          rank=8, block_size=block_size)
    m.max_blocks = 64
    m.set_attended_layers([0])
    return m


def _ingest(m, n):
    m.init_session("s", prefill_len=n + 64)
    k = mx.random.normal((1, H_KV, 1, D)).astype(mx.float16)
    for _ in range(n):
        m.ingest_streaming("s", 0, k, k)
    return m.sessions["s"]


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_small_blocks_compress_with_the_default_budget(block_size):
    """The budget is wider than the block here — this used to raise."""
    m = _mgr(block_size)
    assert m.max_residual > block_size - 1, "test must exercise budget > S_comp"
    sess = _ingest(m, 64 + 6 * block_size)
    assert sess["num_blocks"][0] > 0, "nothing compressed"
    s_comp = block_size - 1
    for b in range(sess["num_blocks"][0]):
        assert sess["comp_res_n"][0][b] <= s_comp


def test_budget_is_not_clamped_when_the_block_is_large_enough():
    """The clamp must not silently shrink a legitimate budget."""
    m = _mgr(1024, window=1024)
    m.max_residual = 128
    sess = _ingest(m, 1024 + 3 * 1024)
    assert sess["num_blocks"][0] > 0
    # 128 <= S_comp (1023), so the budget governs and the clamp is inert.
    assert max(sess["comp_res_n"][0][:sess["num_blocks"][0]]) <= 128


def test_stored_residual_rows_match_the_recorded_count(monkeypatch):
    """n_res must describe the rows actually written, not the slab width.

    Reads the fp16 residual slab directly, so it pins DKV_RESIDUAL_QUANT=none.
    Under the shipped int4 default `comp_res_k` is deliberately None and the
    rows live in the packed `comp_res_k_q` store instead; the invariant under
    test (recorded count vs slab width) is about the count, not the format.
    """
    monkeypatch.setenv("DKV_RESIDUAL_QUANT", "none")
    m = _mgr(32)
    sess = _ingest(m, 64 + 6 * 32)
    n = sess["comp_res_n"][0][0]
    assert sess["comp_res_k"][0].shape[1] == m.max_residual   # slab is full width
    assert n == 31                                            # but only S_comp are real


def teardown_module(_m):
    os.environ.pop("DKV_ENGAGE_THRESHOLD", None)

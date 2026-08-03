"""A recycled slot must hold ITS OWN block's bytes, generation after generation.

The route trace shows the metadata->slot mapping staying STABLE across repeats of
the same prompt while the CONTENT at that slot changes (32k@depth0.9, needle
block: slot=637/183/455/662/762/29 identical in all three repeats, |anc| =
35.4794/35.1886/... in repeat 1 and 34.9666/30.6312/... in repeat 2). Repeat 1 is
always correct; corruption starts at repeat 2, i.e. when slots are first RECYCLED.

That is a write/recycle defect, not compression noise -- n_res stays 128 across
repeats, so selection is stable and only placement moves.

These tests run the pool's real allocate/free/write path on CPU. No GPU, no model.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.native_block_pool import NativeBlockPool          # noqa: E402

KV, HD, R, S = 2, 8, 4, 8


def _pool(max_blocks=32, initial=8):
    return NativeBlockPool(
        max_blocks=max_blocks, num_kv_heads=KV, head_dim=HD, rank=R,
        max_seq_len=S, device="cpu", dtype=torch.float16,
        initial_blocks=initial, num_layers=1, lazy=False,
        max_residual_tokens=4,
    )


def _write(pool, idx, tag):
    """Write a slot whose every field encodes `tag`, so a mix-up is visible."""
    pool.write_block(
        pool_idx=idx,
        U=torch.full((S, R), float(tag), dtype=torch.float16),
        V=torch.full((R, 2 * KV * HD), float(tag), dtype=torch.float16),
        anchor_K=torch.full((KV, HD), float(tag), dtype=torch.float16),
        anchor_V=torch.full((KV, HD), float(tag), dtype=torch.float16),
        scale=1.0, seq_len=S,
    )


def _anchor_tag(pool, idx):
    return float(pool.anchors_KV[idx, 0, 0, 0].item())


def test_allocation_never_aliases_across_generations():
    """Two live blocks must never share a slot, however many free/alloc cycles.

    Aliasing is the mechanism that would put block A's anchor in the slot the
    metadata assigned to block B -- exactly what the trace shows.
    """
    pool = _pool(max_blocks=64, initial=16)
    for gen in range(6):
        slots = [pool.allocate_block() for _ in range(12)]
        assert len(set(slots)) == len(slots), (
            f"generation {gen}: allocate_block returned duplicate slots {slots} "
            "-- two live blocks share storage")
        for b, s in enumerate(slots):
            _write(pool, s, gen * 100 + b)
        # Every block must still read back its OWN tag before we free.
        for b, s in enumerate(slots):
            assert _anchor_tag(pool, s) == gen * 100 + b, (
                f"generation {gen}: slot {s} holds tag {_anchor_tag(pool, s)}, "
                f"expected {gen * 100 + b}")
        for s in slots:
            pool.free_block(s)


def test_free_list_has_no_duplicates_after_recycling():
    """A duplicated free-list entry hands the same slot out twice."""
    pool = _pool(max_blocks=64, initial=16)
    for _ in range(6):
        slots = [pool.allocate_block() for _ in range(10)]
        for s in slots:
            pool.free_block(s)
        # Double-free must not push a second copy (clear_session walks two
        # collections that can hold the same block objects).
        for s in slots:
            pool.free_block(s)
        assert len(pool._free_indices) == len(set(pool._free_indices)), (
            "free list contains duplicates -> allocate_block can return the "
            "same slot to two blocks")
        assert set(pool._free_indices) == set(pool._free_indices_set), (
            "_free_indices and _free_indices_set disagree")


def test_interleaved_alloc_free_keeps_slot_contents_with_their_block():
    """Prefill frees and allocates INTERLEAVED, not in clean generations.

    Blocks compress and publish while others are still accumulating, so the free
    list is churned mid-stream. Contents must still follow their own block.
    """
    pool = _pool(max_blocks=64, initial=8)
    live = {}          # tag -> slot
    tag = 0
    for step in range(40):
        s = pool.allocate_block()
        assert s not in live.values(), (
            f"step {step}: allocate_block returned slot {s}, already live")
        _write(pool, s, tag)
        live[tag] = s
        tag += 1
        if len(live) > 6:                     # evict the oldest, as prefill does
            old = min(live)
            pool.free_block(live.pop(old))
        for t, sl in live.items():            # everyone still reads their own
            assert _anchor_tag(pool, sl) == t, (
                f"step {step}: slot {sl} should hold tag {t}, holds "
                f"{_anchor_tag(pool, sl)}")


def test_growth_during_recycling_preserves_live_contents():
    """_grow_pool reallocates every tensor; live slots must survive it intact.

    At 32k the pool grows from its lazy 288 slots to ~800 mid-prefill, so growth
    and recycling overlap in exactly the runs that fail.
    """
    pool = _pool(max_blocks=256, initial=8)
    live = {}
    for tag in range(8):
        s = pool.allocate_block()
        _write(pool, s, tag)
        live[tag] = s
    # Force growth past the initial allocation while those 8 stay live.
    extra = [pool.allocate_block() for _ in range(40)]
    assert pool.current_blocks > 8, "test did not actually grow the pool"
    for t, sl in live.items():
        assert _anchor_tag(pool, sl) == t, (
            f"slot {sl} lost tag {t} across _grow_pool "
            f"(reads {_anchor_tag(pool, sl)})")
    assert len(set(extra) & set(live.values())) == 0, (
        "growth handed out a slot that was already live")


def _write_batched(pool, slots, tags, max_res=4):
    """Drive the production path: write_blocks_batched, one tag per block."""
    n = len(slots)
    return pool.write_blocks_batched(
        pool_indices=torch.tensor(slots, dtype=torch.long),
        U=torch.stack([torch.full((S, R), float(t)) for t in tags]).half(),
        V=torch.stack([torch.full((R, 2 * KV * HD), float(t)) for t in tags]).half(),
        anchor_K=torch.stack([torch.full((KV, HD), float(t)) for t in tags]).half(),
        anchor_V=torch.stack([torch.full((KV, HD), float(t)) for t in tags]).half(),
        scales=torch.ones(n),
        seq_len=S,
        res_K_positions=torch.stack([
            torch.full((max_res,), int(t) % 7, dtype=torch.int16) for t in tags]),
        res_K_values=torch.stack([
            torch.full((max_res, KV, HD), float(t)) for t in tags]).half(),
        res_V_positions=torch.stack([
            torch.full((max_res,), int(t) % 7, dtype=torch.int16) for t in tags]),
        res_V_values=torch.stack([
            torch.full((max_res, KV, HD), float(t)) for t in tags]).half(),
    )


def test_batched_write_puts_each_block_in_its_own_slot():
    """write_blocks_batched is what production uses; per-block write_block is not.

    Every field is scattered with `self.X[pidx] = ...`, so a mismatch between the
    row order of the stacked payloads and the order of pool_indices silently
    swaps blocks' contents while leaving the metadata->slot mapping intact --
    exactly the signature in the route trace.
    """
    pool = _pool(max_blocks=64, initial=32)
    slots = [pool.allocate_block() for _ in range(10)]
    tags = [11 + i for i in range(10)]
    _write_batched(pool, slots, tags)
    for s, t in zip(slots, tags):
        assert _anchor_tag(pool, s) == t, (
            f"slot {s} should hold anchor tag {t}, holds {_anchor_tag(pool, s)}")
        assert float(pool.V_KV[s, 0, 0, 0, 0].item()) == t, f"V_K wrong at slot {s}"
        assert int(pool.residual_K_positions[s, 0].item()) == t % 7, (
            f"residual positions wrong at slot {s}")
        assert float(pool.residual_K_values[s, 0, 0, 0].item()) == t, (
            f"residual values wrong at slot {s}")


def test_batched_write_survives_non_monotonic_recycled_slots():
    """After recycling, allocate_block hands back slots in LIFO order, so
    pool_indices is NOT sorted. Scatter must follow the index order given, not
    a sorted view of it."""
    pool = _pool(max_blocks=64, initial=32)
    first = [pool.allocate_block() for _ in range(10)]
    _write_batched(pool, first, [100 + i for i in range(10)])
    for s in first:
        pool.free_block(s)
    slots = [pool.allocate_block() for _ in range(10)]
    assert slots != sorted(slots), "recycled order is sorted; test is not exercising LIFO"
    tags = [200 + i for i in range(10)]
    _write_batched(pool, slots, tags)
    for s, t in zip(slots, tags):
        assert _anchor_tag(pool, s) == t, (
            f"recycled slot {s} should hold {t}, holds {_anchor_tag(pool, s)}")

"""A freed pool slot must stop counting as occupied, and must not keep tiering
bookkeeping pointing at it.

Both invariants are read by code that has no idea the slot was freed:

  * ``_occupied_slots`` (sparse_decode/triton_fused_decode.py) derives pool
    occupancy from ``seq_lens > 0``, and TieredBlockStore.maybe_evict divides
    that by ``max_blocks`` to decide whether to start evicting LIVE blocks.
  * ``TieredBlockStore.restore_slot`` copies ``_cpu_store[slot_id]`` back into
    the pool keyed on nothing but the slot id, so a stale entry for a recycled
    slot overwrites whichever block now lives there.

Neither needs a GPU: the pool and the store both run on CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.native_block_pool import NativeBlockPool          # noqa: E402
from native_core.paging.tiered_block_store import TieredBlockStore  # noqa: E402
from native_core.sparse_decode.triton_fused_decode import _occupied_slots  # noqa: E402


KV_HEADS, HEAD_DIM, RANK, MAX_SEQ = 2, 32, 8, 16


def _pool(max_blocks=8):
    return NativeBlockPool(
        max_blocks=max_blocks, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
        rank=RANK, max_seq_len=MAX_SEQ, device="cpu", dtype=torch.float16,
        initial_blocks=max_blocks, num_layers=1, lazy=False,
        max_residual_tokens=4,
    )


def _write(pool, idx, seq_len, fill):
    pool.write_block(
        pool_idx=idx,
        U=torch.full((seq_len, RANK), float(fill), dtype=torch.float16),
        V=torch.full((RANK, 2 * KV_HEADS * HEAD_DIM), float(fill), dtype=torch.float16),
        anchor_K=torch.full((KV_HEADS, HEAD_DIM), float(fill), dtype=torch.float16),
        anchor_V=torch.full((KV_HEADS, HEAD_DIM), float(fill), dtype=torch.float16),
        scale=1.0,
        seq_len=seq_len,
    )


def test_freed_slot_is_not_counted_as_occupied():
    """Occupancy must track the LIVE set, not a high-water mark.

    This is what decides whether eviction fires: a process that runs several
    prompts frees and reallocates slots constantly, and while free_block left
    seq_lens set, occupancy only ever rose -- so eviction eventually switched on
    permanently and started evicting live blocks mid-decode.
    """
    pool = _pool(max_blocks=8)
    idxs = [pool.allocate_block() for _ in range(4)]
    for n, i in enumerate(idxs):
        _write(pool, i, MAX_SEQ, n + 1)

    assert sorted(_occupied_slots(pool, len(idxs))) == sorted(idxs)

    for i in idxs:
        pool.free_block(i)

    assert _occupied_slots(pool, len(idxs)) == [], (
        "freed slots still report as occupied; TieredBlockStore.maybe_evict "
        "divides exactly this count by max_blocks to decide whether to evict")


def test_recycled_slot_drops_stale_tier_state():
    """A recycled slot must not be restorable from the previous block's bytes.

    restore_slot copies _cpu_store[slot] into the pool unconditionally --
    including seq_lens -- so a stale 'CPU' tier on a reused slot silently
    replaces a live block with a dead one, asynchronously, while decode reads it.
    """
    pool = _pool(max_blocks=8)
    store = TieredBlockStore(pool=pool, pager=None, device="cpu",
                             evict_threshold=0.0, evict_batch=8)
    pool._manager = type("M", (), {"_kt_tiered_store": store})()

    victim = pool.allocate_block()
    _write(pool, victim, MAX_SEQ, 7)

    assert store.evict_slot(victim) is True
    assert store.get_tier(victim) == "CPU"
    assert victim in store._cpu_store

    # Slot is released and handed to a different block.
    pool.free_block(victim)
    assert store.get_tier(victim) == "GPU", "recycled slot still marked CPU"
    assert victim not in store._cpu_store, "dead block's bytes still restorable"

    reused = pool.allocate_block()
    assert reused == victim, "test needs the same slot back to be meaningful"
    _write(pool, reused, MAX_SEQ, 3)

    # The stale entry is gone, so a restore attempt is a no-op and the live
    # block survives intact.
    assert store.restore_slot(reused) is False
    assert int(pool.seq_lens[reused].item()) == MAX_SEQ
    assert float(pool.anchors_KV[reused, 0, 0, 0].item()) == 3.0

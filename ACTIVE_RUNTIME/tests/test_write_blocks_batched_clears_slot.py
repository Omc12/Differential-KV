"""write_blocks_batched must CLEAR a recycled slot, exactly as write_block does.

The batched writer's docstring claims it "Mirrors write_block() field-for-field",
and for U / V_KV / residuals / anchors / desc it does. It did NOT clear the
stratified group -- U_sem, U_sem_scale, U_fact, n_semantic -- nor the fact
anchors, so a slot inherited those four fields from whatever block previously
occupied it.

That is not inert on CUDA. _needs_legacy_slots is hardcoded True
(native_block_pool.py:121, with the `not (_is_cuda_dev and _gpu_compress)` form
commented out immediately above), so the tensors are allocated on exactly the
path that uses the batched writer.

Why it hid for so long: a FRESH slot is zeros from torch.zeros, so the first
prompt in a process is always clean and every single-prompt test passes. Only a
RECYCLED slot carries the stale split, so it takes two different prompts in one
process to see it -- which is why the observed signature was "first prompt
correct, later prompts garbage" and why it vanished entirely under
DKV_NO_SLOT_REUSE=1.

test_write_blocks_batched_parity.py cannot catch this: it writes both pools from
clean, where "cleared" and "never written" are indistinguishable. The recycling
is the whole test.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.native_block_pool import NativeBlockPool  # noqa: E402


def _pool(device):
    return NativeBlockPool(
        max_blocks=8, num_kv_heads=2, head_dim=16, rank=4, max_seq_len=8,
        device=device, dtype=torch.float16, num_layers=1, lazy=False,
        max_residual_tokens=2,
    )


def _write_batched(pool, slot, fill, device):
    n, s, r = 1, 8, 4
    kv, hd = 2, 16
    pool.write_blocks_batched(
        pool_indices=torch.tensor([slot], device=device, dtype=torch.long),
        U=torch.full((n, s, r), fill, device=device, dtype=torch.float32),
        V=torch.full((n, r, 2 * kv * hd), fill, device=device, dtype=torch.float32),
        anchor_K=torch.full((n, kv, hd), fill, device=device, dtype=torch.float32),
        anchor_V=torch.full((n, kv, hd), fill, device=device, dtype=torch.float32),
        scales=torch.ones((n,), device=device, dtype=torch.float32),
        seq_len=s,
    )


def test_batched_write_clears_stratified_fields_on_a_recycled_slot():
    """A slot's stratified group must not survive into its next occupant."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = _pool(device)
    if pool.n_semantic is None:
        pytest.skip("pool built without stratified slots")

    slot = 0

    # Occupant 1: dirty the stratified group the way write_block would for a
    # block that HAS a semantic split.
    _write_batched(pool, slot, 1.0, device)
    pool.n_semantic[slot] = 5
    pool.U_sem[slot] = 7
    pool.U_sem_scale[slot] = 3.0
    pool.U_fact[slot] = 2.0

    # Occupant 2: the slot is freed and rewritten by a different block that has
    # NO semantic split. Nothing it passes mentions the stratified group, so if
    # the writer does not clear, occupant 1's split is still sitting there.
    pool.free_block(slot)
    _write_batched(pool, slot, -1.0, device)

    assert int(pool.n_semantic[slot]) == 0, (
        f"n_semantic survived recycling: {int(pool.n_semantic[slot])} != 0 — the "
        "new block will be split at the previous occupant's boundary"
    )
    assert torch.count_nonzero(pool.U_sem[slot]) == 0, "U_sem survived recycling"
    assert torch.count_nonzero(pool.U_sem_scale[slot]) == 0, "U_sem_scale survived"
    assert torch.count_nonzero(pool.U_fact[slot]) == 0, "U_fact survived recycling"


def test_fresh_slot_is_clean_so_a_one_prompt_test_cannot_catch_this():
    """Pins WHY this needed a two-prompt repro, so the next reader doesn't
    'simplify' the test above back into a single-write one."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = _pool(device)
    if pool.n_semantic is None:
        pytest.skip("pool built without stratified slots")

    _write_batched(pool, 3, 1.0, device)
    # Never previously occupied -> zeros regardless of whether the writer clears.
    assert int(pool.n_semantic[3]) == 0
    assert torch.count_nonzero(pool.U_sem[3]) == 0

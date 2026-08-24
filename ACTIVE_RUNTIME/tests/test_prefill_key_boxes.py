"""The prefill router's per-block key boxes: growth folding and rotation.

Two things here are easy to get wrong and impossible to notice from a recall
result, because a WRONG box still produces a plausible ranking:

  * the box is now built INCREMENTALLY as a block accumulates -- only the rows
    added since the cached box are measured, and the result is folded with it.
    A fold that drops rows silently narrows the box, which makes the router's
    upper bound on q.k too low and can drop the one block that mattered.
  * on an UNROTATED pool the keys are rotated at their own absolute positions
    before the reduction, so the box lands in the same frame as the post-RoPE
    query. Rotating at the wrong positions is exactly as silent.

Both are pinned against a direct, obvious reference: build the same box in one
shot and compare.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime.dkv_attention import _apply_rope_single, _prefill_block_key_boxes

H_KV, D, ROT = 2, 16, 16


class _Blk:
    """The subset of a streaming block the box builder reads."""

    def __init__(self, anchor_idx, anchor, active):
        self.anchor_idx = anchor_idx
        self.anchor_kv = anchor.reshape(1, 1, H_KV, D)
        self.active_k = None if active is None else active.unsqueeze(0)
        self.state = "ACCUMULATING"


def _mk(anchor_idx, n, seed):
    g = torch.Generator().manual_seed(seed)
    anchor = torch.randn(H_KV, D, generator=g)
    active = torch.randn(H_KV, n, D, generator=g) if n else None
    return anchor, active


def _reference(anchor, active):
    keys = anchor.unsqueeze(1) if active is None else torch.cat(
        [anchor.unsqueeze(1), active], dim=1)
    return keys.amin(dim=1), keys.amax(dim=1)


def test_box_matches_a_one_shot_reduction():
    anchor, active = _mk(0, 12, seed=1)
    b = _Blk(0, anchor, active)
    mn, mx = _prefill_block_key_boxes([b], torch.device("cpu"))
    rmn, rmx = _reference(anchor, active)
    assert torch.allclose(mn[0], rmn) and torch.allclose(mx[0], rmx)


def test_growth_fold_equals_a_full_rebuild():
    """A block measured in stages must end up with the same box as one measured
    once at its final length. This is the assertion the incremental path exists
    to satisfy, and the one a dropped-rows bug would break."""
    anchor, active = _mk(0, 40, seed=2)
    b = _Blk(0, anchor, active[:, :8])
    dev = torch.device("cpu")
    _prefill_block_key_boxes([b], dev)                 # first measurement
    for n in (17, 29, 40):                             # grows, re-measured each time
        b.active_k = active[:, :n].unsqueeze(0)
        mn, mx = _prefill_block_key_boxes([b], dev)
    rmn, rmx = _reference(anchor, active)
    assert torch.allclose(mn[0], rmn), (mn[0] - rmn).abs().max()
    assert torch.allclose(mx[0], rmx), (mx[0] - rmx).abs().max()


def test_growth_fold_never_narrows_the_box():
    """The failure mode with teeth: a box that is too TIGHT makes the router's
    upper bound on q.k too low, so a block can be ranked below the cutoff on
    evidence that was never measured. A box that is too wide only costs speed."""
    anchor, active = _mk(0, 32, seed=3)
    dev = torch.device("cpu")
    staged = _Blk(0, anchor, active[:, :5])
    _prefill_block_key_boxes([staged], dev)
    staged.active_k = active.unsqueeze(0)
    smn, smx = _prefill_block_key_boxes([staged], dev)
    rmn, rmx = _reference(anchor, active)
    assert (smn[0] <= rmn + 1e-6).all(), "staged box is narrower than the truth"
    assert (smx[0] >= rmx - 1e-6).all(), "staged box is narrower than the truth"


def _rope(max_pos):
    g = torch.Generator().manual_seed(7)
    ang = torch.rand(max_pos, ROT, generator=g) * 6.28
    return torch.cos(ang).unsqueeze(0), torch.sin(ang).unsqueeze(0)


def test_rotated_box_uses_each_key_s_own_absolute_position():
    """anchor -> anchor_idx, active row j -> anchor_idx + 1 + j.

    Compared against rotating the keys independently at those positions and
    reducing. An off-by-one in the position mapping passes every shape check and
    every smoke test, and shows up only as worse routing.
    """
    a0, n = 37, 11
    anchor, active = _mk(a0, n, seed=4)
    b = _Blk(a0, anchor, active)
    dev = torch.device("cpu")
    mn, mx = _prefill_block_key_boxes([b], dev, rope=_rope)

    cos, sin = _rope(a0 + 1 + n)
    pos = torch.tensor([a0] + [a0 + 1 + j for j in range(n)])
    keys = torch.cat([anchor.unsqueeze(1), active], dim=1)         # [H_kv, 1+n, D]
    keys = _apply_rope_single(keys, cos[0, pos].unsqueeze(0),
                              sin[0, pos].unsqueeze(0))
    assert torch.allclose(mn[0], keys.amin(dim=1), atol=1e-5)
    assert torch.allclose(mx[0], keys.amax(dim=1), atol=1e-5)


def test_rotated_and_unrotated_boxes_do_not_share_a_cache_entry():
    """The two live in different frames. Serving one to a reader expecting the
    other is silent, so the cache probe carries a rotation marker."""
    a0 = 5
    anchor, active = _mk(a0, 9, seed=5)
    b = _Blk(a0, anchor, active)
    dev = torch.device("cpu")
    plain_mn, _ = _prefill_block_key_boxes([b], dev)
    rot_mn, _ = _prefill_block_key_boxes([b], dev, rope=_rope)
    again_mn, _ = _prefill_block_key_boxes([b], dev)
    assert not torch.allclose(plain_mn[0], rot_mn[0]), (
        "rotation did not change the box; the rope path is not running")
    assert torch.allclose(plain_mn[0], again_mn[0]), (
        "the unrotated box came back rotated -- the cache probe is not "
        "distinguishing the two frames")


def test_growth_fold_is_consistent_under_rotation_too():
    anchor, active = _mk(11, 24, seed=6)
    dev = torch.device("cpu")
    b = _Blk(11, anchor, active[:, :6])
    _prefill_block_key_boxes([b], dev, rope=_rope)
    b.active_k = active.unsqueeze(0)
    mn, mx = _prefill_block_key_boxes([b], dev, rope=_rope)

    one = _Blk(11, anchor, active)
    rmn, rmx = _prefill_block_key_boxes([one], dev, rope=_rope)
    assert torch.allclose(mn[0], rmn[0], atol=1e-5)
    assert torch.allclose(mx[0], rmx[0], atol=1e-5)


# ── The unrotated-pool opt-in (DKV_SPARSE_PREFILL_ROTATE) ────────────────────


def _blocks(n_blocks, per=40, seed=9):
    out = []
    for i in range(n_blocks):
        anchor, active = _mk(i * (per + 1), per, seed=seed + i)
        out.append(_Blk(i * (per + 1), anchor, active))
    return out


@pytest.mark.parametrize("flag,expect_selective", [("0", False), ("1", True)])
def test_rotate_optin_gates_routing_on_an_unrotated_pool(monkeypatch, flag,
                                                         expect_selective):
    """OFF by default, and the default must be attend-all.

    The capability is real -- 616 of 868 calls become selective at 32k -- but on
    an unrotated pool it measured as no resolvable throughput change while
    costing first-token KL 0.00024 -> 0.00585, so it does not ship on. This pins
    both halves: the flag must actually change behaviour, and unset must be the
    old attend-all.
    """
    from runtime import dkv_attention as DA

    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.setenv("DKV_SPARSE_PREFILL", "1")
    monkeypatch.setenv("DKV_SPARSE_PREFILL_ROTATE", flag)
    # Set the router's geometry explicitly. With the shipped defaults these
    # synthetic blocks span 1230 tokens, which is below DKV_SPARSE_PREFILL_MIN
    # (2048) and inside the 1024-token recency window -- so the router would
    # decline for reasons that have nothing to do with the flag under test, and
    # the parametrised "on" case would fail while the code was correct.
    monkeypatch.setenv("DKV_SPARSE_PREFILL_MIN", "0")
    monkeypatch.setenv("DKV_SPARSE_PREFILL_WINDOW", "41")
    monkeypatch.setattr(DA, "_pool_rotated_k", lambda: False)

    blocks = _blocks(30)
    # chunk_q is [B, H_q, T, D] -- the scorer takes chunk_q[0] and derives
    # gpk = H_q // H_kv from it. H_q == H_kv here, so gpk == 1.
    q = torch.randn(1, H_KV, 4, D)
    kept = DA._sparse_prefill_filter_blocks(
        blocks, q, sink_blocks=1, chunk_start=30 * 41, rope=_rope)
    if expect_selective:
        assert len(kept) < len(blocks), (
            "DKV_SPARSE_PREFILL_ROTATE=1 did not route; the opt-in is dead")
    else:
        assert len(kept) == len(blocks), (
            "routing ran with the opt-in unset -- the default must be attend-all")


def test_rotate_optin_is_inert_without_a_rope_callback(monkeypatch):
    """A caller that cannot supply positions must degrade to attend-all, never
    to scoring a post-RoPE query against pre-RoPE keys."""
    from runtime import dkv_attention as DA

    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.setenv("DKV_SPARSE_PREFILL_ROTATE", "1")
    monkeypatch.setenv("DKV_SPARSE_PREFILL_MIN", "0")
    monkeypatch.setenv("DKV_SPARSE_PREFILL_WINDOW", "41")
    monkeypatch.setattr(DA, "_pool_rotated_k", lambda: False)
    blocks = _blocks(30)
    q = torch.randn(1, H_KV, 4, D)
    kept = DA._sparse_prefill_filter_blocks(blocks, q, sink_blocks=1,
                                            chunk_start=30 * 41, rope=None)
    assert len(kept) == len(blocks)

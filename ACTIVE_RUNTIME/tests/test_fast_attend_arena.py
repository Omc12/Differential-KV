"""attend_fast and the incremental decode arena (2026-11-24).

attend_fast must be the same attention as attend_with_remat (same rows, same
validity, same scale); the arena must hold exactly the routed set, rebuild only
new or rewritten blocks, and never serve a recycled slot's old content.
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from native_core.sparse_decode.remat_cache import (  # noqa: E402
    attend_fast, attend_with_remat, to_head_major)
from native_core.sparse_decode.block_arena import (  # noqa: E402
    BlockArena, bump_slot_epoch)

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("groups,H", [(1, 2), (6, 2), (4, 4)])
def test_attend_fast_matches_sdpa_path(dev, groups, H):
    torch.manual_seed(0)
    dt = torch.float16 if dev == "cuda" else torch.float32
    N, S1, D = 5, 17, 32
    K = torch.randn(N, S1, H, D, device=dev, dtype=dt)
    V = torch.randn_like(K)
    seq = torch.tensor([16, 16, 7, 16, 3], device=dev)
    K[2, 9:] = 1e3                      # padding garbage must not be attended
    K[4, 5:] = 1e3
    dk = torch.randn(1, H, 11, D, device=dev, dtype=dt)
    dv = torch.randn_like(dk)
    ck = (torch.randn(1, H, 1, D, device=dev, dtype=dt),
          torch.randn(1, H, 1, D, device=dev, dtype=dt))
    q = torch.randn(1, H * groups, 1, D, device=dev, dtype=dt)
    ref = attend_with_remat(q, K, V, seq, dk, dv, 11, groups, curr_kv=ck,
                            attn_scale=0.3)
    Kt, Vt, valid = to_head_major(K, V, seq)
    out = attend_fast(q, Kt, Vt, valid, dk, dv, 11, groups, curr_kv=ck,
                      attn_scale=0.3)
    assert out.shape == ref.shape
    tol = 3e-3 if dt == torch.float16 else 1e-5
    assert torch.allclose(out.float(), ref.float(), atol=tol), \
        (out.float() - ref.float()).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dtype path")
def test_attend_fast_bf16_model_fp16_pool():
    """granite NF4: query and window in bf16, materialised blocks in fp16."""
    torch.manual_seed(1)
    dev = "cuda"
    N, S1, H, D, G = 3, 9, 2, 32, 4
    K = torch.randn(N, S1, H, D, device=dev, dtype=torch.float16)
    V = torch.randn_like(K)
    seq = torch.tensor([8, 8, 8], device=dev)
    dk = torch.randn(1, H, 5, D, device=dev, dtype=torch.bfloat16)
    dv = torch.randn_like(dk)
    q = torch.randn(1, H * G, 1, D, device=dev, dtype=torch.bfloat16)
    ref = attend_with_remat(q, K, V, seq, dk, dv, 5, G, attn_scale=0.2)
    Kt, Vt, valid = to_head_major(K, V, seq)
    out = attend_fast(q, Kt, Vt, valid, dk, dv, 5, G, attn_scale=0.2)
    assert out.dtype == q.dtype
    assert torch.allclose(out.float(), ref.float(), atol=2e-2)


class _Pool:
    """Minimal pool: slot content is a deterministic function of (slot, epoch)."""

    def __init__(self, n_slots, H=2, D=8, S1=5):
        self.U = torch.zeros(n_slots, S1 - 1, 4)
        self.content = {s: 0 for s in range(n_slots)}
        self.H, self.D, self.S1 = H, D, S1
        self.calls = []

    def rewrite(self, slot):
        self.content[slot] += 1
        bump_slot_epoch(self, torch.tensor([slot]))

    def materialise(self, bi, ai):
        slots = bi.tolist()
        self.calls.append(slots)
        K = torch.stack([torch.full((self.S1, self.H, self.D), float(s * 100 + self.content[s]))
                         for s in slots])
        V = K + 0.5
        return K, V, {"seq_lens": torch.full((len(slots),), self.S1 - 1)}


def _arena_slots(ar):
    """Multiset of slot values the arena would attend (one per valid block)."""
    Kv, _, valid = ar.view()
    rows = Kv[0, :, 0][valid]
    vals = sorted(set(rows.tolist()))
    return vals


def test_arena_reuses_kept_blocks_and_rebuilds_only_new():
    p = _Pool(10)
    ar = BlockArena()
    assert ar.refresh(p, torch.tensor([1, 2, 3]), None, 32, p.materialise)
    assert sorted(p.calls[-1]) == [1, 2, 3]
    assert _arena_slots(ar) == [100.0, 200.0, 300.0]
    assert ar.refresh(p, torch.tensor([2, 3, 4]), None, 32, p.materialise)
    assert p.calls[-1] == [4]                    # only the new block was built
    assert _arena_slots(ar) == [200.0, 300.0, 400.0]
    n_calls = len(p.calls)
    assert ar.refresh(p, torch.tensor([4, 3, 2]), None, 32, p.materialise)
    assert len(p.calls) == n_calls               # same set: nothing rebuilt


def test_arena_rebuilds_a_rewritten_slot():
    p = _Pool(10)
    ar = BlockArena()
    ar.refresh(p, torch.tensor([1, 2]), None, 32, p.materialise)
    p.rewrite(2)                                  # slot 2 now holds new content
    ar.refresh(p, torch.tensor([1, 2]), None, 32, p.materialise)
    assert p.calls[-1] == [2]
    assert _arena_slots(ar) == [100.0, 201.0]    # never the old 200


def test_arena_grows_shrinks_and_masks_departed_blocks():
    p = _Pool(40)
    ar = BlockArena()
    ar.refresh(p, torch.tensor([1, 2]), None, 32, p.materialise)
    ar.refresh(p, torch.arange(1, 9), None, 32, p.materialise)
    assert _arena_slots(ar) == [float(s * 100) for s in range(1, 9)]
    ar.refresh(p, torch.tensor([5]), None, 32, p.materialise)
    assert _arena_slots(ar) == [500.0]
    # A rank change rebuilds everything.
    n = len(p.calls)
    ar.refresh(p, torch.tensor([5]), None, 64, p.materialise)
    assert len(p.calls) == n + 1 and p.calls[-1] == [5]


def test_arena_declines_duplicate_slots():
    p = _Pool(10)
    ar = BlockArena()
    assert not ar.refresh(p, torch.tensor([1, 1]), None, 32, p.materialise)

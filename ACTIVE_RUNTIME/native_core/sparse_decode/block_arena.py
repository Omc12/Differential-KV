"""Incremental refresh of the materialised routed blocks (CUDA decode).

The remat cache rebuilt EVERY routed block on every refresh (every
DKV_REMAT_INTERVAL tokens, and on any pool write to the layer): gather the
factors, U @ V, residual scatter, RoPE at each row's position. That cost grows
with K, and it is most of what made a large routed set expensive.

Between two refreshes the routed set barely changes. The arena keeps each
block's materialised rows, head-major ([H_kv, rows, D], the layout
remat_cache.attend_fast reads), across refreshes and rebuilds only:

  * blocks that are new to the routed set, and
  * blocks whose pool slot was rewritten since they were materialised.

Attention over a set of rows does not depend on their order, so a new block
simply takes a free position; positions of blocks that left the set are masked
until reused. Refresh cost then follows routing churn, not K.

Slot rewrites are tracked by a per-slot write epoch on the pool, bumped by
native_block_pool's write_block / write_blocks_batched / free_block (the same
three places that invalidate the router's key boxes). A block is reused only
when its slot's epoch still matches the one it was built from, so a recycled
slot can never be read as its previous content.

DKV_REMAT_ARENA=0 restores the full rebuild.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

import torch


def arena_enabled() -> bool:
    return os.environ.get("DKV_REMAT_ARENA", "1") == "1"


# ── per-slot write epochs ────────────────────────────────────────────────────

def _pool_capacity(pool) -> int:
    U = getattr(pool, "U", None)
    return int(U.shape[0]) if U is not None else 0


def _epoch_table(pool, create: bool) -> Optional[torch.Tensor]:
    t = getattr(pool, "slot_epoch", None)
    cap = _pool_capacity(pool)
    if t is None:
        if not create or cap == 0:
            return None
        t = torch.zeros(cap, dtype=torch.long, device=pool.U.device)
        pool.slot_epoch = t
    elif t.numel() < cap:
        # The pool grew: new slots start at epoch 0 (never materialised).
        t = torch.cat([t, torch.zeros(cap - t.numel(), dtype=t.dtype, device=t.device)])
        pool.slot_epoch = t
    return t


def bump_slot_epoch(pool, slots) -> None:
    """Mark slots as rewritten. No-op until an arena has read the pool."""
    t = _epoch_table(pool, create=False)
    if t is None:
        return
    if isinstance(slots, torch.Tensor):
        idx = slots.to(device=t.device, dtype=torch.long).reshape(-1)
    else:
        idx = torch.tensor([int(slots)], device=t.device, dtype=torch.long)
    if idx.numel():
        t.index_add_(0, idx, torch.ones_like(idx))


def slot_epochs(pool, slots: torch.Tensor) -> torch.Tensor:
    t = _epoch_table(pool, create=True)
    return t[slots.to(device=t.device, dtype=torch.long)]


# ── the arena ────────────────────────────────────────────────────────────────

class BlockArena:
    """One layer's materialised routed blocks, kept across refreshes."""

    def __init__(self) -> None:
        self.key = None
        self.reset()

    def reset(self) -> None:
        self.Kt: Optional[torch.Tensor] = None     # [H, cap*S1, D]
        self.Vt: Optional[torch.Tensor] = None
        self.valid: Optional[torch.Tensor] = None  # [cap*S1] bool
        self.S1 = 0
        self.rank = None
        self.pos_slot: List[int] = []              # -1 = free
        self.pos_epoch: List[int] = []
        self.used = 0
        self.rebuilt = 0                            # blocks materialised (stats)
        self.reused = 0

    @property
    def cap(self) -> int:
        return len(self.pos_slot)

    def _rows(self, positions: List[int], device) -> torch.Tensor:
        p = torch.tensor(positions, device=device, dtype=torch.long)
        return (p.view(-1, 1) * self.S1
                + torch.arange(self.S1, device=device).view(1, -1)).reshape(-1)

    def _grow(self, new_cap: int) -> None:
        H, _, D = self.Kt.shape
        rows = new_cap * self.S1
        K2 = self.Kt.new_zeros(H, rows, D)
        V2 = self.Vt.new_zeros(H, rows, D)
        m2 = self.valid.new_zeros(rows)
        n = self.cap * self.S1
        K2[:, :n] = self.Kt
        V2[:, :n] = self.Vt
        m2[:n] = self.valid
        self.Kt, self.Vt, self.valid = K2, V2, m2
        self.pos_slot += [-1] * (new_cap - self.cap)
        self.pos_epoch += [-1] * (new_cap - len(self.pos_epoch))

    def refresh(self, pool, block_indices: torch.Tensor,
                anchor_indices: Optional[torch.Tensor], rank,
                materialise: Callable) -> bool:
        """Bring the arena to exactly `block_indices`. False = not handled.

        `materialise(bi, ai)` returns (K [n,S1,H,D], V, gather) for a subset,
        the same call the full rebuild makes for all of them.
        """
        from native_core.sparse_decode.remat_cache import to_head_major

        bi = block_indices.reshape(-1).long()
        n = int(bi.numel())
        if n == 0:
            return False
        ep = slot_epochs(pool, bi)
        slots, epochs = torch.stack([bi.to(ep.device), ep]).tolist()   # one sync
        if len(set(slots)) != n:
            return False                       # duplicates: leave to the full path
        want = dict(zip(slots, epochs))

        if rank != self.rank:
            self.reset()                       # the layer's rank changed: rebuild
            self.rank = rank
        # Too fragmented (the set shrank a lot): rebuild compactly.
        if self.cap and self.used > 2 * n + 8:
            self.reset()
            self.rank = rank

        kept = {}
        for p, (s, e) in enumerate(zip(self.pos_slot, self.pos_epoch)):
            if s >= 0 and want.get(s) == e:
                kept[s] = p
        missing = [i for i, s in enumerate(slots) if s not in kept]
        stale = [p for p, s in enumerate(self.pos_slot)
                 if s >= 0 and kept.get(s) != p]
        free = sorted(set(range(self.cap)) - set(kept.values()))

        if missing:
            idx = torch.tensor(missing, device=bi.device, dtype=torch.long)
            ai = anchor_indices.reshape(-1)[idx.to(anchor_indices.device)] \
                if anchor_indices is not None else None
            K, V, g = materialise(bi[idx].to(block_indices.dtype), ai)
            Kt_new, Vt_new, val_new = to_head_major(K, V, g["seq_lens"])
            S1 = int(K.shape[1])
            if self.Kt is None or S1 != self.S1:
                if self.Kt is not None:             # block width changed: start over
                    self.reset()
                    self.rank = rank
                    return self.refresh(pool, block_indices, anchor_indices,
                                        rank, materialise)
                self.S1 = S1
                H, _, D = Kt_new.shape
                self.Kt = Kt_new.new_zeros(H, 0, D)
                self.Vt = Vt_new.new_zeros(H, 0, D)
                self.valid = val_new.new_zeros(0)
                free = []
            if len(missing) > len(free):
                old = self.cap
                self._grow(old + (len(missing) - len(free)))
                free = free + list(range(old, self.cap))
            targets = free[:len(missing)]
            rows = self._rows(targets, self.Kt.device)
            self.Kt.index_copy_(1, rows, Kt_new.to(self.Kt.dtype))
            self.Vt.index_copy_(1, rows, Vt_new.to(self.Vt.dtype))
            self.valid.index_copy_(0, rows, val_new)
            for t, i in zip(targets, missing):
                self.pos_slot[t] = slots[i]
                self.pos_epoch[t] = epochs[i]
            filled = set(targets)
            self.rebuilt += len(missing)
        else:
            filled = set()
        self.reused += len(kept)

        # Blocks that left the set (or went stale) and were not overwritten.
        drop = [p for p in stale if p not in filled]
        if drop:
            self.valid.index_fill_(0, self._rows(drop, self.valid.device), False)
            for p in drop:
                self.pos_slot[p] = -1
                self.pos_epoch[p] = -1
        occ = [p for p, s in enumerate(self.pos_slot) if s >= 0]
        self.used = (max(occ) + 1) if occ else 0
        return self.used > 0

    def view(self):
        r = self.used * self.S1
        return self.Kt[:, :r], self.Vt[:, :r], self.valid[:r]

"""Router-owned block summaries: exact per-channel key boxes (Quest-style).

WHY THIS EXISTS
    The decode router used to score a compressed block from the keys its STORE
    chose to keep: the anchor plus the residual rows. Router quality therefore
    depended on store quality, and a store that keeps no residual keys (codes)
    still had to keep residual POSITIONS just to feed the router. That coupling
    is what the store/router split is supposed to prevent.

    Here the router owns its own summary. When a block is compressed, the
    elementwise min and max of its EXACT post-RoPE keys are recorded per KV head
    ([H_kv, D] each, 2 x H_kv x D fp16 per block, e.g. 4 KB on granite-8B)
    before the store discards anything. The score

        bound(block) = max over query heads of  sum_d max(q_d * min_d, q_d * max_d)
                     = q . mid + |q| . half,   mid = (max+min)/2, half = (max-min)/2

    is an UPPER BOUND on q.k for every key in the block, so a block holding the
    best-matching key can never score below a block that does not contain a key
    at least that good. It reads 2 vectors per block instead of 1 + R residual
    rows, so routing cost no longer grows with the residual budget.

FRAME
    Boxes are always measured in the POST-RoPE frame, at each key's own absolute
    position (anchor = anchor_idx, active row j = anchor_idx + 1 + j). The
    decode query is post-RoPE, so no rotation happens at scoring time. When the
    pool already stores post-RoPE keys, they are used as they are.

SLOTS
    The table is indexed by pool slot. Any write to a slot invalidates its box
    (native_block_pool.write_block / write_blocks_batched / free_block), and only
    the GPU compressor records one after its write. A slot written by any other
    path therefore has no box; the router treats it as always relevant (+inf),
    which can only attend more, never drop a block.

    DKV_ROUTE_BOX=0 restores the residual-key router.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch


def box_routing_enabled() -> bool:
    return os.environ.get("DKV_ROUTE_BOX", "1").strip() != "0"


class KeyBoxTable:
    """Slot-indexed [cap, H_kv, D] min/max boxes plus a validity mask."""

    def __init__(self, device):
        self.device = device
        self.mn: Optional[torch.Tensor] = None
        self.mx: Optional[torch.Tensor] = None
        self.valid: Optional[torch.Tensor] = None
        self.n_recorded = 0          # host-side: has ANY box ever been recorded

    def _ensure(self, cap: int, H: int, D: int) -> None:
        if self.mn is None or tuple(self.mn.shape[1:]) != (H, D):
            n = max(64, cap)
            self.mn = torch.zeros((n, H, D), device=self.device, dtype=torch.float16)
            self.mx = torch.zeros((n, H, D), device=self.device, dtype=torch.float16)
            self.valid = torch.zeros((n,), device=self.device, dtype=torch.bool)
            return
        old = self.mn.shape[0]
        if cap <= old:
            return
        n = max(cap, 2 * old)
        for name in ("mn", "mx", "valid"):
            t = getattr(self, name)
            nt = torch.zeros((n,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
            nt[:old] = t
            setattr(self, name, nt)

    def set(self, slots: List[int], mn: torch.Tensor, mx: torch.Tensor) -> None:
        if not slots:
            return
        self._ensure(max(slots) + 1, int(mn.shape[1]), int(mn.shape[2]))
        idx = torch.tensor(slots, device=self.device, dtype=torch.long)
        self.mn[idx] = mn.to(self.device, torch.float16)
        self.mx[idx] = mx.to(self.device, torch.float16)
        self.valid[idx] = True
        self.n_recorded += len(slots)

    def invalidate(self, slots) -> None:
        if self.valid is None:
            return
        if isinstance(slots, int):
            if 0 <= slots < self.valid.shape[0]:
                self.valid[slots] = False
            return
        idx = torch.as_tensor(slots, device=self.device).long().reshape(-1)
        idx = idx[(idx >= 0) & (idx < self.valid.shape[0])]
        if idx.numel():
            self.valid[idx] = False

    def gather(self, slots_long: torch.Tensor):
        """-> (mn [N,H,D], mx [N,H,D], valid [N]) for these slots; None if empty."""
        if self.mn is None or self.n_recorded == 0:
            return None
        cap = self.mn.shape[0]
        s = slots_long.to(self.device).long()
        inb = s < cap
        sc = torch.where(inb, s, torch.zeros_like(s))
        return self.mn[sc], self.mx[sc], self.valid[sc] & inb


def table_for(pool) -> Optional[KeyBoxTable]:
    return getattr(pool, "key_box_table", None)


def invalidate_slots(pool, slots) -> None:
    t = getattr(pool, "key_box_table", None)
    if t is not None:
        t.invalidate(slots)


def _rotary_for(manager):
    """The model's rotary forward, or None if keys must not / cannot be rotated.

    Returns ("none", None) when the pool already stores post-RoPE keys (use the
    keys as they are), ("rope", fn) to rotate, or (None, None) when the frame
    cannot be established -- then no box is recorded at all.
    """
    try:
        from native_core.sparse_decode.triton_fused_decode import pool_stores_rotated_k
        if pool_stores_rotated_k():
            return "none", None
    except Exception:                                            # noqa: BLE001
        return None, None
    model = getattr(manager, "model", None) if manager is not None else None
    if model is None:
        return None, None
    try:
        from runtime.dkv_attention import _resolve_rotary_emb
        fn = _resolve_rotary_emb(model)
    except Exception:                                            # noqa: BLE001
        fn = None
    return ("rope", fn) if fn is not None else (None, None)


def exact_key_boxes(blocks_list, manager, chunk: int = 16
                    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Post-RoPE (min, max) over each block's exact keys -> two [N, H_kv, D].

    Must run while `active_k` still holds the block's keys (before the store
    encodes and drops them). All blocks share one T (the compressor's batch
    contract). Returns None if the frame cannot be established.
    """
    if not blocks_list:
        return None
    mode, rot = _rotary_for(manager)
    if mode is None:
        return None
    from runtime.dkv_attention import _apply_rope_single
    mins, maxs = [], []
    for s in range(0, len(blocks_list), chunk):
        grp = blocks_list[s:s + chunk]
        anc = torch.stack([b.anchor_kv[0, 0] for b in grp], dim=0)      # [g, H, D]
        act = torch.cat([b.active_k for b in grp], dim=0)               # [g, H, T, D]
        keys = torch.cat([anc.unsqueeze(2).to(act.dtype), act], dim=2)  # [g, H, 1+T, D]
        if mode == "rope":
            n = keys.shape[2]
            a0 = torch.tensor([int(getattr(b, "anchor_idx", 0) or 0) for b in grp],
                              device=keys.device, dtype=torch.long)
            pos = a0.unsqueeze(1) + torch.arange(n, device=keys.device).unsqueeze(0)  # [g, n]
            cos, sin = rot(keys[:, 0], pos)                             # [g, n, rot]
            keys = _apply_rope_single(keys, cos.unsqueeze(1).to(keys.dtype),
                                      sin.unsqueeze(1).to(keys.dtype))
        mins.append(keys.amin(dim=2))
        maxs.append(keys.amax(dim=2))
        del keys, act, anc
    return torch.cat(mins, 0), torch.cat(maxs, 0)


def record(pool, blocks_list, boxes) -> None:
    """After the pool write: attach boxes to the slots and to the blocks."""
    if pool is None or boxes is None:
        return
    mn, mx = boxes
    t = getattr(pool, "key_box_table", None)
    if t is None:
        t = KeyBoxTable(mn.device)
        pool.key_box_table = t
    slots = [int(b.pool_idx) for b in blocks_list]
    t.set(slots, mn, mx)
    for i, b in enumerate(blocks_list):
        try:
            b._exact_key_box = (mn[i], mx[i])
        except Exception:                                        # noqa: BLE001
            pass


def box_relevance(Q: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor,
                  valid: torch.Tensor, scale: float) -> torch.Tensor:
    """Upper-bound score per block -> [N]. Q is [H, L, D] post-RoPE.

    Max over query heads and query tokens; a block with no recorded box scores
    +inf (always routed).
    """
    H, L, D = Q.shape
    H_kv = mn.shape[1]
    gpk = max(1, H // H_kv)
    q = Q.reshape(H_kv, gpk * L, D).float()                             # [k, gL, D]
    mid = ((mx.float() + mn.float()) * 0.5).permute(1, 2, 0)            # [k, D, N]
    half = ((mx.float() - mn.float()) * 0.5).permute(1, 2, 0)
    bound = torch.bmm(q, mid) + torch.bmm(q.abs(), half)                # [k, gL, N]
    rel = bound.amax(dim=(0, 1)) * float(scale)                         # [N]
    return torch.where(valid, rel, torch.full_like(rel, float("inf")))

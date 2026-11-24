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

import math
import os
from typing import List, Optional, Tuple

import torch


def box_routing_enabled() -> bool:
    return os.environ.get("DKV_ROUTE_BOX", "1").strip() != "0"


def adaptive_routing_enabled() -> bool:
    """DKV_ROUTE_ADAPTIVE: choose HOW MANY blocks per query by estimated mass.

    See select_adaptive. Needs the moments recorded at compression, so it
    must be on when the prompt is compressed, not only at decode.
    """
    return os.environ.get("DKV_ROUTE_ADAPTIVE", "1").strip() != "0"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class KeyBoxTable:
    """Slot-indexed [cap, H_kv, D] min/max boxes plus a validity mask.

    Also holds each block's key MOMENTS for the adaptive router (mean and
    per-channel variance of the exact post-RoPE keys, and log of the token
    count). They share the box's slot validity: a write invalidates both.
    """

    def __init__(self, device):
        self.device = device
        self.mn: Optional[torch.Tensor] = None
        self.mx: Optional[torch.Tensor] = None
        self.mu: Optional[torch.Tensor] = None      # [cap, H, D] fp16
        self.var: Optional[torch.Tensor] = None     # [cap, H, D] fp32 (can exceed fp16)
        self.logn: Optional[torch.Tensor] = None    # [cap] fp32
        self.valid: Optional[torch.Tensor] = None
        self.mvalid: Optional[torch.Tensor] = None  # moments recorded for the slot
        self.n_recorded = 0          # host-side: has ANY box ever been recorded
        self.n_moments = 0           # host-side: has ANY moment ever been recorded

    _FIELDS = (("mn", torch.float16, True), ("mx", torch.float16, True),
               ("mu", torch.float16, True), ("var", torch.float32, True),
               ("logn", torch.float32, False), ("valid", torch.bool, False),
               ("mvalid", torch.bool, False))

    def _ensure(self, cap: int, H: int, D: int) -> None:
        if self.mn is None or tuple(self.mn.shape[1:]) != (H, D):
            n = max(64, cap)
            for name, dt, hd in self._FIELDS:
                shape = (n, H, D) if hd else (n,)
                setattr(self, name, torch.zeros(shape, device=self.device, dtype=dt))
            return
        old = self.mn.shape[0]
        if cap <= old:
            return
        n = max(cap, 2 * old)
        for name, _, _ in self._FIELDS:
            t = getattr(self, name)
            nt = torch.zeros((n,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
            nt[:old] = t
            setattr(self, name, nt)

    def set(self, slots: List[int], mn: torch.Tensor, mx: torch.Tensor,
            moments: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
            ) -> None:
        if not slots:
            return
        self._ensure(max(slots) + 1, int(mn.shape[1]), int(mn.shape[2]))
        idx = torch.tensor(slots, device=self.device, dtype=torch.long)
        self.mn[idx] = mn.to(self.device, torch.float16)
        self.mx[idx] = mx.to(self.device, torch.float16)
        self.valid[idx] = True
        self.n_recorded += len(slots)
        if moments is not None:
            mu, var, logn = moments
            self.mu[idx] = mu.to(self.device, torch.float16)
            self.var[idx] = var.to(self.device, torch.float32)
            self.logn[idx] = logn.to(self.device, torch.float32)
            self.mvalid[idx] = True
            self.n_moments += len(slots)
        else:
            self.mvalid[idx] = False

    def invalidate(self, slots) -> None:
        if self.valid is None:
            return
        if isinstance(slots, int):
            if 0 <= slots < self.valid.shape[0]:
                self.valid[slots] = False
                self.mvalid[slots] = False
            return
        idx = torch.as_tensor(slots, device=self.device).long().reshape(-1)
        idx = idx[(idx >= 0) & (idx < self.valid.shape[0])]
        if idx.numel():
            self.valid[idx] = False
            self.mvalid[idx] = False

    def _idx(self, slots_long: torch.Tensor):
        cap = self.mn.shape[0]
        s = slots_long.to(self.device).long()
        inb = s < cap
        return torch.where(inb, s, torch.zeros_like(s)), inb

    def gather(self, slots_long: torch.Tensor):
        """-> (mn [N,H,D], mx [N,H,D], valid [N]) for these slots; None if empty."""
        if self.mn is None or self.n_recorded == 0:
            return None
        sc, inb = self._idx(slots_long)
        return self.mn[sc], self.mx[sc], self.valid[sc] & inb

    def gather_moments(self, slots_long: torch.Tensor):
        """-> (mu [N,H,D], var [N,H,D], logn [N], mvalid [N]); None if none recorded."""
        if self.mn is None or self.n_moments == 0:
            return None
        sc, inb = self._idx(slots_long)
        return self.mu[sc], self.var[sc], self.logn[sc], self.mvalid[sc] & inb


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
    mins, maxs, mus, vars_, logns = [], [], [], [], []
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
        if adaptive_routing_enabled():
            kf = keys.float()
            mus.append(kf.mean(dim=2))
            vars_.append(kf.var(dim=2, unbiased=False))
            logns.append(torch.full((kf.shape[0],), math.log(kf.shape[2]),
                                    device=kf.device))
            del kf
        del keys, act, anc
    out = (torch.cat(mins, 0), torch.cat(maxs, 0))
    if mus:
        out = out + ((torch.cat(mus, 0), torch.cat(vars_, 0), torch.cat(logns, 0)),)
    return out


def record(pool, blocks_list, boxes) -> None:
    """After the pool write: attach boxes (and moments) to the slots and blocks."""
    if pool is None or boxes is None:
        return
    mn, mx = boxes[0], boxes[1]
    moments = boxes[2] if len(boxes) > 2 else None
    t = getattr(pool, "key_box_table", None)
    if t is None:
        t = KeyBoxTable(mn.device)
        pool.key_box_table = t
    slots = [int(b.pool_idx) for b in blocks_list]
    t.set(slots, mn, mx, moments)
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


# ── Adaptive block count (DKV_ROUTE_ADAPTIVE) ────────────────────────────────
#
# A fixed K routes the same number of blocks for a lookup that lives in one
# block and for a question whose evidence is spread over forty. The box above
# is an upper bound on a block's BEST key; it says nothing about how much
# softmax mass the block would receive, which is what decides whether leaving
# it out matters. The moments estimate that mass:
#
#     lse_b(q) ~ log n + s q.mu + (s^2 / 2) sum_d q_d^2 var_d
#
# (the log-mean-exp of a Gaussian score with the block's per-channel key mean
# and variance), clipped to the two bounds every block satisfies,
#     log n + s q.mu  <=  lse_b  <=  log n + s * box_bound.
# GROW-ONLY: the fixed K's box top-K is always routed, exactly as before. If
# those blocks hold less than P of the compressed blocks' estimated mass
# (shares averaged over query heads), blocks are added in order of estimated
# share until the total reaches P, up to KMAX. So the rule can only attend
# more than the fixed K, never a different or smaller set.
#
# Offline on captured Qwen3.5-4B RULER KV (benchmarks/route_adaptive_screen.py,
# exact keys, routing error alone), share of the true attention mass covered
# and mean blocks routed:
#
#                       32k (27 blocks)        64k (59 blocks)
#     box top-16        0.935 at 16            0.788 at 16
#     box top-32          -                    0.877 at 32
#     grow-only P=0.80  0.950 at 16.9          0.924 at 32.9
#     grow-only P=0.90  0.975 at 19.7          0.961 at 42.6
#
# At equal blocks the estimate adds the RIGHT blocks (0.924 vs 0.877 at ~32),
# and it barely grows at 32k, where the real-decode check (colab/
# logit_fidelity.py, Qwen2.5-1.5B, 5 prompts) found extra compressed blocks
# cost a little (decode-step KL 0.0041 at 16 blocks, 0.0055 at 20, 0.0058 at
# all 31): their reconstruction error outweighs the mass they add while the
# box's 16 already hold most of it. Hence P=0.80 by default.

def estimate_block_lse(Q: torch.Tensor, mn, mx, mu, var, logn, scale: float):
    """Q [H, L, D] post-RoPE -> estimated log-mass [H*L, N] per query row."""
    H, L, D = Q.shape
    H_kv = mn.shape[1]
    gpk = max(1, H // H_kv)
    q = Q.reshape(H_kv, gpk * L, D).float()                             # [k, gL, D]
    mid = ((mx.float() + mn.float()) * 0.5).permute(1, 2, 0)            # [k, D, N]
    half = ((mx.float() - mn.float()) * 0.5).permute(1, 2, 0)
    ub = torch.bmm(q, mid) + torch.bmm(q.abs(), half)                   # [k, gL, N]
    lin = torch.bmm(q, mu.float().permute(1, 2, 0))                     # [k, gL, N]
    quad = torch.bmm(q * q, var.float().permute(1, 2, 0))               # [k, gL, N]
    s = float(scale)
    ln = logn.float().view(1, 1, -1)
    lo = ln + s * lin
    est = torch.minimum(torch.maximum(lo + 0.5 * s * s * quad, lo), ln + s * ub)
    return est.reshape(H_kv * gpk * L, -1)


def select_adaptive(est: torch.Tensor, box_rel: torch.Tensor, ok: torch.Tensor,
                    k_floor: int) -> torch.Tensor:
    """-> indices into the N candidates: the box's top-k_floor, then additions.

    est [R, N] estimated log-mass per query row; box_rel [N] box score (+inf
    for a block with no box); ok [N] moments recorded. A block with no
    moments or no box is always selected (it can only attend more).
    """
    R, N = est.shape
    P = _env_float("DKV_ROUTE_ADAPTIVE_P", 0.80)
    kmax = max(k_floor, _env_int("DKV_ROUTE_ADAPTIVE_KMAX", 64))
    dev = est.device
    forced = ~ok | ~torch.isfinite(box_rel)
    est = torch.where(ok.view(1, -1), est, torch.full_like(est, float("-inf")))
    share = torch.softmax(est, dim=-1).nan_to_num(0.0).mean(0)          # [N]
    # The fixed router's set: box top-k_floor (box-less blocks rank first).
    floor = torch.zeros(N, dtype=torch.bool, device=dev)
    floor[torch.topk(torch.nan_to_num(box_rel.float(), posinf=1e30),
                     min(k_floor, N)).indices] = True
    keep = floor | forced
    # Grow by estimated share until the routed set holds P of it.
    covered = (share * keep.float()).sum()
    rest = torch.where(keep, torch.full_like(share, -1.0), share)
    srt, order = rest.sort(descending=True)
    cum = covered + srt.clamp(min=0.0).cumsum(0)
    n_add = torch.where(covered >= P, torch.zeros((), dtype=torch.long, device=dev),
                        (cum < P).sum() + 1)
    room = (kmax - keep.sum()).clamp(min=0)
    avail = (srt >= 0).sum()
    n_add = torch.minimum(torch.minimum(n_add, room), avail)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(N, device=dev)
    keep = keep | (rank < n_add)
    # Best first: forced, then the box's set, then additions by share.
    box_rank = torch.zeros(N, device=dev)
    _by_box = torch.topk(torch.nan_to_num(box_rel.float(), posinf=1e30), N).indices
    box_rank[_by_box] = torch.arange(N, 0, -1, device=dev, dtype=torch.float32) / N
    score = torch.where(forced, torch.full_like(share, 4.0),
                        torch.where(floor, 2.0 + box_rank, share))
    score = torch.where(keep, score, torch.full_like(score, -1.0))
    n_sel = int(keep.sum())                               # the one sync per routing call
    return torch.topk(score, n_sel).indices

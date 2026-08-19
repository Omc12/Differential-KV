"""
native_core/compression/basis_group.py

Shared low-rank bases across blocks.

WHAT THIS IS FOR
----------------
Every compressed block currently stores its own basis ``V`` — a
``[rank, 2*kv_heads*head_dim]`` fp16 matrix.  On the `mid` preset
(rank 32, kv_heads 2, head_dim 128, pool block 257) that is

    U        257 * 32 * 1      =   8,224 B
    V_K+V_V   32 * 2*128*2*2   =  32,768 B     <-- 39% of the slot
    anchors      2*128*2*2     =   1,024 B
    residuals 40 * 2*128*2*2   =  40,960 B
                                  --------
                                   83,136 B

``V`` is the largest single item after the residual store, and it is the one
that is most redundant between blocks: adjacent prose blocks from the same
document span nearly the same subspace of key/value space.  If G blocks agree
on a basis, the pool needs one copy instead of G, and the same VRAM budget
holds proportionally more blocks.

THE MATH
--------
A block's delta is stored factored as ``D ~= U V`` with ``U``: [T, k],
``V``: [k, F].  Given a group basis ``Vg``: [r, F] with ORTHONORMAL ROWS, the
best approximation of ``U V`` inside span(Vg) is

    U' = U (V Vg^T)          and         D ~= U' Vg

so re-expressing a block in a shared basis is one [k, F] x [F, r] matmul.  No
re-decomposition, no touching the original K/V.

The energy that survives that projection is the quantity worth thresholding on
— not the raw principal angles, because those weight every basis direction
equally while the block's energy is concentrated in the first few:

    G     = U^T U                       [k, k]
    C     = V Vg^T                      [k, r]
    kept  = tr(G C C^T) / tr(G V V^T)

``kept`` is in [0, 1] and is exactly ``||U' Vg||_F^2 / ||U V||_F^2``.  A block
joins a group when ``kept >= threshold``; the lost ``1 - kept`` of delta energy
is what the shared basis costs.  Blocks that lose too much keep their own
basis, and — when the basis store is full — the ones forced to share are
exactly the ones the anchor-delta residual pass (see ``residual_capture``)
should repair.

Both traces are [k, k], so scoring a block against a group never touches the
token dimension T.

NOTHING HERE IS ENABLED BY DEFAULT.  This module is pure math plus a registry;
the pool and the compress path decide whether to use it (DKV_SHARED_BASIS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

__all__ = [
    "row_orthonormalize",
    "retained_energy",
    "reproject_U",
    "BasisGroup",
    "SharedBasisRegistry",
    "shared_basis_enabled",
    "shared_basis_fraction",
    "shared_basis_threshold",
]


# ── Configuration ────────────────────────────────────────────────────────────

def shared_basis_enabled() -> bool:
    """DKV_SHARED_BASIS — off by default.

    Turning this on changes the POOL LAYOUT (V_KV gets fewer rows than there
    are blocks, reached through an indirection map), so it is opt-in until it
    has been measured on a real model.  Off, every code path below is dead and
    the pool allocates exactly as it did before.
    """
    return os.environ.get("DKV_SHARED_BASIS", "0").strip().lower() not in (
        "0", "", "off", "false", "no")


def shared_basis_fraction() -> float:
    """Basis rows to allocate, as a fraction of block slots.

    0.25 means "budget for 4x sharing": the V store gets ceil(0.25 * n_blocks)
    rows.  This is a CAPACITY CONTRACT, not a prediction — when the store fills,
    blocks force-join their nearest group rather than failing, so a document
    that shares worse than the budget assumed degrades in fidelity, not in
    correctness.  Clamped to (0, 1].
    """
    try:
        f = float(os.environ.get("DKV_SHARED_BASIS_FRAC", "0.25"))
    except ValueError:
        f = 0.25
    return min(1.0, max(1e-3, f))


def shared_basis_threshold() -> float:
    """Minimum retained delta energy for a voluntary join.

    0.90 = "the shared basis must keep 90% of this block's delta energy".
    Blocks below it get their own basis while the store has room.
    """
    try:
        t = float(os.environ.get("DKV_SHARED_BASIS_THRESHOLD", "0.90"))
    except ValueError:
        t = 0.90
    return min(1.0, max(0.0, t))


def _prescreen_width() -> int:
    """How many candidate groups get the exact energy test per block.

    Groups are ranked by a cheap top-direction cosine first; only this many go
    on to the [k,k] trace.  0 disables the prescreen (exact test against every
    group).
    """
    try:
        return max(0, int(os.environ.get("DKV_SHARED_BASIS_PRESCREEN", "8")))
    except ValueError:
        return 8


# ── Core math ────────────────────────────────────────────────────────────────

def row_orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """Return a basis with orthonormal ROWS spanning the same space as V's rows.

    V: [..., k, F] -> [..., k, F].  Implemented as a QR of V^T; degenerate rows
    (rank-deficient V, which happens whenever a block's dynamic_rank truncation
    left zero rows) come back as exact zeros rather than NaN, and a zero row
    contributes nothing to any projection.

    The stored V is NOT orthonormal in general — the DKV_V_SCALE undo divides
    the V-half columns of Vh by a per-block gain after the SVD — so this is a
    real step, not a formality.
    """
    Vf = V.float()
    # QR of V^T: [..., F, k] -> Q [..., F, k] with orthonormal columns.
    Q, R = torch.linalg.qr(Vf.transpose(-1, -2), mode="reduced")
    On = Q.transpose(-1, -2)                                   # [..., k, F]
    # Kill rows whose R diagonal is ~0: those directions were not present in V,
    # and QR fills them with arbitrary complement directions.  Keeping them
    # would let a group claim span it does not actually have.
    diag = torch.diagonal(R, dim1=-2, dim2=-1).abs()           # [..., k]
    tol = diag.amax(dim=-1, keepdim=True) * 1e-6
    On = On * (diag > tol).to(On.dtype).unsqueeze(-1)
    return torch.nan_to_num(On, nan=0.0, posinf=0.0, neginf=0.0)


def retained_energy(
    U: torch.Tensor,        # [N, T, k]
    V: torch.Tensor,        # [N, k, F]
    Vg: torch.Tensor,       # [G, r, F] row-orthonormal
    eps: float = 1e-12,
) -> torch.Tensor:          # [N, G] in [0, 1]
    """Fraction of each block's delta energy that survives projection onto each
    group basis.

    ``kept[n, g] = ||U_n (V_n Vg_g^T) Vg_g||_F^2 / ||U_n V_n||_F^2``

    computed through the [k, k] Gram matrix so the token dimension T is only
    touched once, by the Gram itself.
    """
    if Vg.numel() == 0 or U.numel() == 0:
        return U.new_zeros((U.shape[0], Vg.shape[0] if Vg.dim() == 3 else 0))

    Uf = U.float()
    Vf = V.float()
    Vgf = Vg.float()
    N, _, k = Uf.shape
    G, r, F = Vgf.shape

    Gram = torch.bmm(Uf.transpose(1, 2), Uf)                   # [N, k, k]

    # den[n] = tr(Gram_n V_n V_n^T) = sum_{a,b} Gram[a,b] * <V[b], V[a]>
    VVt = torch.bmm(Vf, Vf.transpose(1, 2))                    # [N, k, k]
    den = (Gram * VVt).sum(dim=(1, 2))                         # [N]

    # C[n, g] = V_n Vg_g^T -> [N, k, G*r] -> [N, G, k, r]
    C = torch.matmul(Vf, Vgf.reshape(G * r, F).t())            # [N, k, G*r]
    C = C.reshape(N, k, G, r).permute(0, 2, 1, 3)              # [N, G, k, r]

    # num[n, g] = tr(Gram_n C C^T) = sum_{a,b} Gram[a,b] * <C[b], C[a]>
    CCt = torch.matmul(C, C.transpose(-1, -2))                 # [N, G, k, k]
    num = (Gram.unsqueeze(1) * CCt).sum(dim=(-1, -2))          # [N, G]

    kept = num / den.clamp(min=eps).unsqueeze(1)
    # A block with no delta energy at all (den ~ 0) is perfectly representable
    # by ANY basis — reconstructing zero is exact — so it should join freely
    # rather than being reported as a total loss.
    kept = torch.where(den.unsqueeze(1) > eps, kept, torch.ones_like(kept))
    return torch.nan_to_num(kept, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def reproject_U(
    U: torch.Tensor,        # [N, T, k]
    V: torch.Tensor,        # [N, k, F]
    Vg: torch.Tensor,       # [N, r, F] row-orthonormal, one basis PER BLOCK
) -> torch.Tensor:          # [N, T, r]
    """Re-express each block's U in its assigned group basis: ``U' = U (V Vg^T)``.

    ``U' Vg`` is then the orthogonal projection of ``U V`` onto span(Vg), i.e.
    the best possible reconstruction from that basis.  Vg is per-block here
    (already gathered by group id), not the [G, r, F] store.
    """
    Cmat = torch.bmm(V.float(), Vg.float().transpose(1, 2))    # [N, k, r]
    Up = torch.bmm(U.float(), Cmat)                            # [N, T, r]
    return torch.nan_to_num(Up, nan=0.0, posinf=0.0, neginf=0.0)


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass
class BasisGroup:
    """One shared basis and its bookkeeping.

    ``row`` is the row index into the pool's V store; it is the identity a
    block records.  ``members`` is a refcount — a group whose last member is
    freed can be reclaimed.
    """
    row: int
    rank: int
    members: int = 0
    layer: int = -1
    # First basis direction, unit-norm, kept on the host for the cheap
    # prescreen.  This is the top right-singular direction of the founding
    # block, so two groups with near-parallel top directions are the pairs
    # worth spending the exact test on.
    top_dir: Optional[torch.Tensor] = None


@dataclass
class Assignment:
    """Result of assigning one block to a basis."""
    row: int            # V-store row this block now reads
    kept: float         # retained delta energy in [0, 1]
    is_new: bool        # True if this block founded the group
    forced: bool        # True if it joined below threshold because the store was full


class SharedBasisRegistry:
    """Greedy online grouping of block bases against a fixed-size basis store.

    Policy, in order:
      1. Score the block against existing groups (prescreened by top-direction
         cosine, then the exact retained-energy test).
      2. If the best group keeps >= threshold, join it.
      3. Otherwise, if the store has a free row, found a new group.
      4. Otherwise FORCE-JOIN the best group.  Capacity is a hard contract;
         running out of basis rows must degrade fidelity, never fail a write.

    Groups are keyed by layer so a block never searches bases belonging to a
    different layer.  Cross-layer subspaces are unrelated in practice and the
    exact test would reject them anyway — this just keeps the search small.
    """

    def __init__(
        self,
        capacity: int,
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.capacity = max(1, int(capacity))
        self.threshold = shared_basis_threshold() if threshold is None else float(threshold)
        self.device = device
        self.dtype = dtype
        self.groups: List[BasisGroup] = []
        self._by_layer: dict = {}
        self._free_rows: List[int] = list(range(self.capacity - 1, -1, -1))
        # Diagnostics — cheap counters, read by tests and the VRAM audit.
        self.n_joined = 0
        self.n_founded = 0
        self.n_forced = 0
        self.kept_sum = 0.0
        self.kept_count = 0

    # -- bookkeeping ---------------------------------------------------------

    def reset(self) -> None:
        self.groups.clear()
        self._by_layer.clear()
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        self.n_joined = self.n_founded = self.n_forced = 0
        self.kept_sum = 0.0
        self.kept_count = 0

    def release_row(self, row: int) -> None:
        """Drop one member from the group occupying ``row``; reclaim it at zero.

        Called when a pool slot is freed.  A reclaimed row goes back on the
        free list, so a long session that cycles through topics does not
        permanently lose basis capacity to groups nobody reads any more.
        """
        for i, g in enumerate(self.groups):
            if g.row != row:
                continue
            g.members -= 1
            if g.members <= 0:
                self.groups.pop(i)
                lst = self._by_layer.get(g.layer)
                if lst:
                    self._by_layer[g.layer] = [x for x in lst if x is not g]
                self._free_rows.append(row)
            return

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def mean_kept(self) -> float:
        return self.kept_sum / self.kept_count if self.kept_count else 1.0

    def stats(self) -> dict:
        return {
            "groups": self.n_groups,
            "capacity": self.capacity,
            "founded": self.n_founded,
            "joined": self.n_joined,
            "forced": self.n_forced,
            "mean_kept": self.mean_kept,
        }

    # -- assignment ----------------------------------------------------------

    def assign_batch(
        self,
        U: torch.Tensor,          # [N, T, k]
        V: torch.Tensor,          # [N, k, F]  (rows NOT assumed orthonormal)
        layer: int,
        basis_store: torch.Tensor,  # [capacity, r, F] — written in place
    ) -> Tuple[List[Assignment], torch.Tensor]:
        """Assign every block in a compress batch to a basis row.

        Returns the per-block assignments and the per-block gathered bases
        ``[N, r, F]`` (row-orthonormal) that ``reproject_U`` should be called
        with.  ``basis_store`` is mutated: newly founded groups write their
        orthonormalised basis into their row.

        Blocks are processed in order, so a block CAN join a group founded by
        an earlier block of the same batch.  That is the common case inside a
        document: consecutive prose blocks share a subspace.
        """
        N = int(U.shape[0])
        r_store = int(basis_store.shape[1])
        F = int(basis_store.shape[2])

        # Orthonormalise every candidate basis once, padded/truncated to the
        # store's row width.  A block with k < r keeps zero rows at the tail;
        # zero rows project nothing, so they are harmless in both directions.
        V_pad = V.new_zeros((N, r_store, F))
        k_use = min(int(V.shape[1]), r_store)
        V_pad[:, :k_use, :] = V[:, :k_use, :]
        V_on = row_orthonormalize(V_pad)                        # [N, r, F]

        assignments: List[Assignment] = []
        gathered = basis_store.new_zeros((N, r_store, F))

        prescreen = _prescreen_width()

        for n in range(N):
            u_n = U[n : n + 1]
            v_n = V[n : n + 1]
            cand = self._by_layer.get(layer, [])

            best_g: Optional[BasisGroup] = None
            best_kept = -1.0

            if cand:
                # Cheap prescreen on the top basis direction.
                sub = cand
                if prescreen and len(cand) > prescreen:
                    top_n = V_on[n, 0]
                    sims = []
                    for g in cand:
                        if g.top_dir is None:
                            sims.append(1.0)                    # untested -> keep
                        else:
                            sims.append(float(torch.dot(top_n, g.top_dir.to(top_n.device)).abs()))
                    order = sorted(range(len(cand)), key=lambda i: -sims[i])[:prescreen]
                    sub = [cand[i] for i in order]

                rows = torch.tensor([g.row for g in sub], device=basis_store.device,
                                    dtype=torch.long)
                kept = retained_energy(u_n, v_n, basis_store[rows])[0]   # [len(sub)]
                j = int(torch.argmax(kept).item())
                best_kept = float(kept[j].item())
                best_g = sub[j]

            if best_g is not None and best_kept >= self.threshold:
                best_g.members += 1
                gathered[n] = basis_store[best_g.row]
                assignments.append(Assignment(best_g.row, best_kept, False, False))
                self.n_joined += 1
                self.kept_sum += best_kept
                self.kept_count += 1
                continue

            if self._free_rows:
                row = self._free_rows.pop()
                basis_store[row] = V_on[n].to(basis_store.dtype)
                g = BasisGroup(row=row, rank=k_use, members=1, layer=layer,
                               top_dir=V_on[n, 0].detach().cpu())
                self.groups.append(g)
                self._by_layer.setdefault(layer, []).append(g)
                gathered[n] = basis_store[row]
                assignments.append(Assignment(row, 1.0, True, False))
                self.n_founded += 1
                self.kept_sum += 1.0
                self.kept_count += 1
                continue

            # Store full and nothing cleared the bar: force-join the best
            # candidate for this layer, or — if this layer has none at all —
            # the best across every layer, so a write can always complete.
            if best_g is None:
                all_rows = torch.tensor([g.row for g in self.groups],
                                        device=basis_store.device, dtype=torch.long)
                kept = retained_energy(u_n, v_n, basis_store[all_rows])[0]
                j = int(torch.argmax(kept).item())
                best_kept = float(kept[j].item())
                best_g = self.groups[j]
            best_g.members += 1
            gathered[n] = basis_store[best_g.row]
            assignments.append(Assignment(best_g.row, best_kept, False, True))
            self.n_forced += 1
            self.kept_sum += best_kept
            self.kept_count += 1

        return assignments, gathered

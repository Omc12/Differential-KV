"""Shared low-rank bases, MLX port of `basis_group.py`.

WHY A SECOND FILE RATHER THAN A BACKEND SWITCH
-----------------------------------------------
`basis_group.py` is torch and is what the CUDA pool drives; it stays the
reference for the MATH, and its 27 tests are the specification. This module is
the `mx` twin, kept separate so the CUDA file is not destabilised by a runtime
it never runs on. `tests/test_basis_group_mlx.py` asserts the two AGREE
numerically rather than asserting the port in isolation -- a port that is
self-consistently wrong passes every test written only against itself.

THE MATH, unchanged from the torch original
--------------------------------------------
A block is stored as ``D ~= U V``. For a group basis ``Vg`` with ORTHONORMAL
ROWS, the best approximation of ``U V`` inside span(Vg) is

    U' = U (V Vg^T)      and      D ~= U' Vg

one ``[k,F] x [F,r]`` matmul -- no re-decomposition, and the original K/V are
never touched. Scoring uses RETAINED ENERGY, not principal angles: angles weight
every basis direction equally while a block's energy sits in the first few. With
``G = U^T U`` and ``C = V Vg^T``,

    kept = tr(G C C^T) / tr(G V V^T)  ==  ||U' Vg||_F^2 / ||U V||_F^2

Both traces are ``[k, k]``, so scoring a block against a group never touches the
token dimension.

MLX-SPECIFIC CONSTRAINT worth knowing before using this
--------------------------------------------------------
``mx.linalg.qr`` is CPU-ONLY -- calling it without an explicit CPU stream raises
"[linalg::qr] This op is not yet supported on the GPU". `row_orthonormalize`
therefore pins a CPU stream. On unified memory that is a scheduling boundary
rather than a copy, but it does serialise, so orthonormalisation happens ONCE
per compress batch (on the founding candidates) and never per block.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx

# The knobs are shared with the torch implementation ON PURPOSE: one set of env
# variables must not mean two different things depending on which runtime read
# them. Importing rather than re-parsing also means a default can only ever be
# changed in one place.
from native_core.compression.basis_group import (  # noqa: F401
    shared_basis_enabled,
    shared_basis_fraction,
    shared_basis_threshold,
    _prescreen_width,
)

_EPS = 1e-12


def row_orthonormalize(V: mx.array) -> mx.array:
    """Basis with orthonormal ROWS spanning the same space as V's rows.

    ``V: [N, k, F] -> [N, k, F]``. QR of ``V^T``; rows that were not actually
    present in V (rank-deficient V, which happens whenever a block's dynamic
    rank truncation left zero rows) come back as exact ZEROS rather than
    arbitrary complement directions. That matters: keeping them would let a
    group claim span it does not have, and a zero row projects nothing.

    The stored V is not orthonormal in general -- the DKV_V_SCALE undo divides
    the V-half columns of Vh by a per-block gain after the SVD -- so this is a
    real step, not a formality.
    """
    Vf = V.astype(mx.float32)
    if Vf.ndim == 2:
        Vf = Vf[None]
        squeeze = True
    else:
        squeeze = False

    # CPU STREAM IS REQUIRED, not a preference: mx.linalg.qr raises on GPU.
    with mx.stream(mx.cpu):
        Q, R = mx.linalg.qr(mx.swapaxes(Vf, -1, -2))     # Q: [N, F, k]
        On = mx.swapaxes(Q, -1, -2)                       # [N, k, F]
        diag = mx.abs(mx.diagonal(R, axis1=-2, axis2=-1))          # [N, k]
        tol = mx.max(diag, axis=-1, keepdims=True) * 1e-6
        On = On * (diag > tol).astype(On.dtype)[..., None]
        On = mx.where(mx.isnan(On) | mx.isinf(On), mx.zeros_like(On), On)
        mx.eval(On)
    return On[0] if squeeze else On


def retained_energy(U: mx.array, V: mx.array, Vg: mx.array) -> mx.array:
    """Fraction of each block's delta energy surviving projection onto each group.

    ``U: [N, T, k]``, ``V: [N, k, F]``, ``Vg: [G, r, F]``.
    Returns ``[N, G]`` in [0, 1].

    Vg is orthonormalised HERE rather than being required orthonormal, because
    the pool stores each group's RAW founding basis (see `reproject_U` for why
    that is load-bearing). The projector onto span(Vg) is the same either way,
    so this changes nothing about the score -- it just moves the requirement off
    the caller.
    """
    if Vg.size == 0 or U.size == 0:
        return mx.zeros((U.shape[0], Vg.shape[0] if Vg.ndim == 3 else 0))
    Vg = row_orthonormalize(Vg)

    Uf = U.astype(mx.float32)
    Vf = V.astype(mx.float32)
    Vgf = Vg.astype(mx.float32)
    N, _, k = Uf.shape
    G, r, F = Vgf.shape

    Gram = mx.matmul(mx.swapaxes(Uf, 1, 2), Uf)                    # [N, k, k]

    # den[n] = tr(Gram_n V_n V_n^T)
    VVt = mx.matmul(Vf, mx.swapaxes(Vf, 1, 2))                     # [N, k, k]
    den = mx.sum(Gram * VVt, axis=(1, 2))                          # [N]

    # C[n, g] = V_n Vg_g^T
    C = mx.matmul(Vf, mx.swapaxes(Vgf.reshape(G * r, F), 0, 1))    # [N, k, G*r]
    C = mx.transpose(C.reshape(N, k, G, r), (0, 2, 1, 3))          # [N, G, k, r]

    CCt = mx.matmul(C, mx.swapaxes(C, -1, -2))                     # [N, G, k, k]
    num = mx.sum(Gram[:, None] * CCt, axis=(-1, -2))               # [N, G]

    kept = num / mx.maximum(den, _EPS)[:, None]
    # A block with no delta energy at all is perfectly representable by ANY
    # basis -- reconstructing zero is exact -- so it joins freely rather than
    # being reported as a total loss.
    kept = mx.where(den[:, None] > _EPS, kept, mx.ones_like(kept))
    kept = mx.where(mx.isnan(kept), mx.zeros_like(kept), kept)
    return mx.clip(kept, 0.0, 1.0)


def reproject_U(U: mx.array, V: mx.array, Vg: mx.array) -> mx.array:
    """``U' = U V Vg^+``, so ``U' Vg`` is the projection of ``U V`` onto span(Vg).

    ``Vg`` is PER BLOCK here (``[N, r, F]``, already gathered by group id), not
    the ``[G, r, F]`` store, and it is NOT assumed to have orthonormal rows.

    THE PSEUDO-INVERSE IS NOT PEDANTRY -- it is what makes a FOUNDER EXACT.
    With ``Vg == V`` (a block that founded its own group, which is every block
    when nothing shares) the right pseudo-inverse gives ``V V^+ == I`` and
    therefore ``U' == U`` and ``U' Vg == U V`` bit-for-bit. The simpler
    ``U (V Vg^T)`` only does that when Vg's rows are ORTHONORMAL, and the joint
    ``[K | V]`` basis this pool stores is NOT: measured row norms 0.78-0.83,
    because the two halves are sliced out of one orthonormal Vh and the V half
    is then divided by the per-block v_scale gain.
    That mattered in production, not in theory. Storing a unit-normalised basis
    and pushing the scale into U leaves ``U V`` unchanged, so reconstruction
    stays exact -- but the ROUTER reads U and V separately, so its per-block
    scores shift and it retains a different set of blocks. The signature was a
    needle that passed at depth 0.0 and failed at 0.5 and 0.9 (depth-dependent
    is routing; depth-invariant would have been reconstruction).

    Trap 3 from the port file lives here: ``r_proj`` can be NARROWER than the
    store rank, because it is ``min(max_rank + oversamples, T_active, feat_dim)``
    and a short block makes the SVD narrower than the pool. ``U'`` has one
    column per BASIS direction, so it does NOT fit back into the ``[N, T,
    r_proj]`` buffer it came from -- the caller must rebuild at the STORE width,
    which is ``Vg.shape[1]`` and is what this returns.
    """
    Vf = V.astype(mx.float32)
    Gf = Vg.astype(mx.float32)
    Cmat = mx.matmul(Vf, mx.swapaxes(Gf, 1, 2))                    # [N, k, r]
    Gram = mx.matmul(Gf, mx.swapaxes(Gf, 1, 2))                    # [N, r, r]
    # Ridge, because zero basis rows (a rank truncation left them) make Gram
    # singular. It is tiny next to the row norms it regularises, and a zero row
    # contributes nothing either way.
    r = Gram.shape[-1]
    Gram = Gram + mx.eye(r, dtype=mx.float32) * 1e-6
    with mx.stream(mx.cpu):                       # solve is CPU-only, like qr
        Cp = mx.linalg.solve(mx.swapaxes(Gram, 1, 2), mx.swapaxes(Cmat, 1, 2))
        mx.eval(Cp)
    Up = mx.matmul(U.astype(mx.float32), mx.swapaxes(Cp, 1, 2))    # [N, T, r]
    return mx.where(mx.isnan(Up) | mx.isinf(Up), mx.zeros_like(Up), Up)


@dataclass
class BasisGroup:
    """One shared basis and its bookkeeping.

    ``row`` is the index into the pool's V store -- the identity a block
    records. ``members`` is a refcount, so a group whose last member is freed
    can be reclaimed.

    NOTE ON TRAP 1, which is NOT solved here. The registry only ever sees blocks
    that actually founded or joined a group, so every group it holds is real.
    The trap lives one level up, in the POOL's slot -> row map: an UNWRITTEN
    slot must still resolve to a valid basis row (row 0) or any gather over it
    reads out of bounds, and "points at row 0" is then indistinguishable from
    "holds a refcount on row 0". Releasing a claim never made would decrement
    the founding block's group, return the row to the free list while blocks are
    still reading it, and let a later block re-found it with a different basis --
    silently changing what earlier blocks decompress to. The pool integration
    must therefore carry an explicit per-slot `claimed` flag and only call
    `release_row` for slots that hold one.
    """
    row: int
    rank: int
    members: int = 0
    layer: int = -1
    top_dir: Optional[mx.array] = None


@dataclass
class Assignment:
    row: int            # V-store row this block now reads
    kept: float         # retained delta energy in [0, 1]
    is_new: bool        # founded the group
    forced: bool        # joined below threshold because the store was full


class SharedBasisRegistryMLX:
    """Greedy online grouping against a fixed-size basis store.

    Policy, in order:
      1. Score the block against existing groups (top-direction prescreen, then
         the exact retained-energy test).
      2. Best group keeps >= threshold -> join it.
      3. Else a free row exists -> found a new group.
      4. Else FORCE-JOIN the best group. Capacity is a HARD CONTRACT: running
         out of basis rows must degrade fidelity, never fail a write.

    Groups are keyed by layer so a block never searches another layer's bases.

    Trap 6 from the port file is why `layer` is normalised by the CALLER with an
    explicit None check rather than `getattr(b, "layer_idx", -1) or -1`: that
    idiom makes layer 0 falsy, so every layer-0 block reports -1 and shares a
    search space with genuinely-unknown blocks. Grouping still "works" -- it
    just groups the wrong set, which is invisible in every aggregate statistic.
    """

    def __init__(self, capacity: int, threshold: Optional[float] = None,
                 dtype: mx.Dtype = mx.float16):
        self.capacity = max(1, int(capacity))
        self.threshold = shared_basis_threshold() if threshold is None else float(threshold)
        self.dtype = dtype
        self.groups: List[BasisGroup] = []
        self._by_layer: dict = {}
        self._free_rows: List[int] = list(range(self.capacity - 1, -1, -1))
        self.n_joined = 0
        self.n_founded = 0
        self.n_forced = 0
        self.kept_sum = 0.0
        self.kept_count = 0

    def reset(self) -> None:
        self.groups.clear()
        self._by_layer.clear()
        self._free_rows = list(range(self.capacity - 1, -1, -1))
        self.n_joined = self.n_founded = self.n_forced = 0
        self.kept_sum = 0.0
        self.kept_count = 0

    def release_row(self, row: int) -> None:
        """Drop one member from the group at ``row``; reclaim the row at zero.

        Trap 2: a slot being OVERWRITTEN must give up its previous claim BEFORE
        the new assignment, or the write decrements the group it just joined.
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
        """Diagnostics.

        Trap 7: do NOT report the config back as a measurement. `sharing_factor`
        computed over pool CAPACITY yields exactly ``1/frac`` whenever the store
        is full, regardless of how many blocks were written -- so it is derived
        here from rows that actually HOLD a claim.

        `joined == 0` is the 4-bit-KV signature: on a q4_0 preset quantisation
        noise takes voluntary joins to zero and retained energy to 0.685 at
        IDENTICAL pool MB, because the saving comes from allocating fewer basis
        rows rather than from grouping succeeding. A run can therefore look like
        a clean win while fidelity has collapsed -- always read `joined` and
        `mean_kept` next to the memory number.
        """
        live = sum(1 for g in self.groups if g.members > 0)
        members = sum(g.members for g in self.groups)
        return {
            "groups": self.n_groups,
            "capacity": self.capacity,
            "founded": self.n_founded,
            "joined": self.n_joined,
            "forced": self.n_forced,
            "mean_kept": self.mean_kept,
            "live_rows": live,
            "sharing_factor": (members / live) if live else 1.0,
        }

    def assign_batch(self, U: mx.array, V: mx.array, layer: int,
                     basis_store: mx.array) -> Tuple[List[Assignment], mx.array]:
        """Assign every block in a compress batch to a basis row.

        Returns the per-block assignments and the gathered per-block bases
        ``[N, r, F]`` to hand to `reproject_U`. A block whose `Assignment` has
        ``is_new`` set FOUNDED its own group, so its gathered basis IS its own V
        and the caller should leave U alone entirely rather than round-tripping
        it through a solve -- see `_assign_shared_basis`.

        Blocks are processed IN ORDER, so a block can join a group founded by an
        earlier block of the same batch -- the common case inside one document,
        where consecutive prose blocks share a subspace.

        `basis_store` is MUTATED IN PLACE, and the caller sees it -- mx
        `__setitem__` has the same aliasing semantics as torch here, verified
        rather than assumed (`test_store_is_mutated_in_place`). Newly founded
        groups write their orthonormalised basis into their row. Nothing needs
        rebinding, which is what makes this a drop-in for the torch call shape.
        """
        N = int(U.shape[0])
        r_store = int(basis_store.shape[1])
        F = int(basis_store.shape[2])

        k_use = min(int(V.shape[1]), r_store)
        V_pad = mx.zeros((N, r_store, F), dtype=mx.float32)
        V_pad[:, :k_use, :] = V[:, :k_use, :].astype(mx.float32)
        V_on = row_orthonormalize(V_pad)                          # [N, r, F]

        assignments: List[Assignment] = []
        gathered = mx.zeros((N, r_store, F), dtype=basis_store.dtype)
        prescreen = _prescreen_width()

        for n in range(N):
            u_n = U[n:n + 1]
            v_n = V[n:n + 1]
            cand = self._by_layer.get(layer, [])

            best_g: Optional[BasisGroup] = None
            best_kept = -1.0

            if cand:
                sub = cand
                if prescreen and len(cand) > prescreen:
                    top_n = V_on[n, 0]
                    sims = [1.0 if g.top_dir is None
                            else float(mx.abs(mx.sum(top_n * g.top_dir)).item())
                            for g in cand]
                    order = sorted(range(len(cand)), key=lambda i: -sims[i])[:prescreen]
                    sub = [cand[i] for i in order]

                rows = mx.array([g.row for g in sub], dtype=mx.int32)
                kept = retained_energy(u_n, v_n, mx.take(basis_store, rows, axis=0))[0]
                j = int(mx.argmax(kept).item())
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
                # RAW, not orthonormalised. A founder must be reproducible
                # EXACTLY, and `reproject_U`'s pseudo-inverse gives U' == U only
                # when the stored basis is the block's own V. Storing a
                # unit-normalised basis keeps U V exact but rescales U and V
                # against each other, which the router -- reading them
                # separately -- turns into a different set of retained blocks.
                basis_store[row] = V_pad[n].astype(basis_store.dtype)
                self.groups.append(BasisGroup(row=row, rank=k_use, members=1,
                                              layer=layer, top_dir=V_on[n, 0]))
                self._by_layer.setdefault(layer, []).append(self.groups[-1])
                gathered[n] = basis_store[row]
                assignments.append(Assignment(row, 1.0, True, False))
                self.n_founded += 1
                self.kept_sum += 1.0
                self.kept_count += 1
                continue

            # Store full and nothing cleared the bar. Force-join the best
            # candidate for this layer, or -- if this layer has none at all --
            # the best across every layer, so a write can always complete.
            if best_g is None:
                all_rows = mx.array([g.row for g in self.groups], dtype=mx.int32)
                kept = retained_energy(u_n, v_n, mx.take(basis_store, all_rows, axis=0))[0]
                j = int(mx.argmax(kept).item())
                best_kept = float(kept[j].item())
                best_g = self.groups[j]
            best_g.members += 1
            gathered[n] = basis_store[best_g.row]
            assignments.append(Assignment(best_g.row, best_kept, False, True))
            self.n_forced += 1
            self.kept_sum += best_kept
            self.kept_count += 1

        mx.eval(basis_store, gathered)
        return assignments, gathered

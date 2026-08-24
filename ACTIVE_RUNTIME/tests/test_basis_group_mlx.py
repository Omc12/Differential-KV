"""The MLX shared-basis math must AGREE WITH THE TORCH REFERENCE, not just with itself.

`basis_group.py` (torch) is the specification -- it is what the CUDA pool runs
and it has 27 tests of its own. A port tested only against its own outputs can
be self-consistently wrong and pass everything, so almost every test here feeds
IDENTICAL inputs to both implementations and asserts the numbers match.

The exceptions are the two properties that are true mathematically rather than
by reference: orthonormal rows, and kept == 1.0 when a block's own basis is the
group basis. Those are asserted directly, because a bug that moved BOTH
implementations the same way would slip past an agreement test.
"""
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_core.compression import basis_group as T          # noqa: E402
from native_core.compression import basis_group_mlx as M      # noqa: E402

N, TT, K, F, R = 6, 24, 4, 16, 4


def _pair(shape, seed):
    """One array, as both a torch tensor and an mx array -- same bytes."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(a), mx.array(a)


def test_row_orthonormalize_matches_torch_span():
    """Rows must be orthonormal and span the same space as the input's rows.

    QR is only unique up to sign, so the two backends may return different
    SIGNED rows for the same input. Comparing rows elementwise would be a false
    failure. What must agree is the SUBSPACE, which the projector V^T V pins
    uniquely -- so that is what is compared.
    """
    vt, vm = _pair((N, K, F), 1)
    ot = T.row_orthonormalize(vt)
    om = M.row_orthonormalize(vm)

    # (a) rows really are orthonormal in the mx port
    gram = np.array(mx.matmul(om, mx.swapaxes(om, -1, -2)))
    np.testing.assert_allclose(gram, np.eye(K)[None].repeat(N, 0), atol=2e-5)

    # (b) same subspace as torch, compared through the sign-invariant projector
    pt = torch.bmm(ot.transpose(1, 2), ot).numpy()
    pm = np.array(mx.matmul(mx.swapaxes(om, 1, 2), om))
    np.testing.assert_allclose(pt, pm, atol=2e-5)


def test_rank_deficient_rows_come_back_as_exact_zeros():
    """A dynamic-rank truncation leaves zero rows; QR would otherwise fill them
    with arbitrary complement directions, letting a group claim span it does not
    have."""
    a = np.random.default_rng(2).standard_normal((1, K, F)).astype(np.float32)
    a[0, 2:] = 0.0                                   # rank 2 of K
    om = M.row_orthonormalize(mx.array(a))
    tail = np.array(om)[0, 2:]
    assert np.abs(tail).max() == 0.0, (
        "rank-deficient rows are not exactly zero -- a zero row must project "
        "nothing, or the group claims span the founding block never had")
    assert not np.isnan(np.array(om)).any()


def test_retained_energy_matches_torch():
    ut, um = _pair((N, TT, K), 3)
    vt, vm = _pair((N, K, F), 4)
    gt, gm = _pair((3, R, F), 5)
    gt, gm = T.row_orthonormalize(gt), M.row_orthonormalize(gm)

    kt = T.retained_energy(ut, vt, gt).numpy()
    km = np.array(M.retained_energy(um, vm, gm))
    np.testing.assert_allclose(kt, km, atol=2e-5)
    assert ((km >= 0.0) & (km <= 1.0)).all()


def test_a_blocks_own_basis_retains_everything():
    """kept == 1.0 against your own orthonormalised basis. Asserted directly
    rather than by agreement: a bug moving both backends identically would
    survive a pure agreement test."""
    _, um = _pair((1, TT, K), 6)
    _, vm = _pair((1, K, F), 7)
    own = M.row_orthonormalize(vm)
    kept = float(M.retained_energy(um, vm, own)[0, 0].item())
    assert kept == pytest.approx(1.0, abs=1e-4), f"kept {kept} against own basis"


def test_zero_energy_block_joins_freely():
    """Reconstructing zero is exact, so a block with no delta energy must score
    1.0 rather than 0.0 -- otherwise it is reported as a total loss and is
    force-joined for no reason."""
    um = mx.zeros((1, TT, K))
    _, vm = _pair((1, K, F), 8)
    _, gm = _pair((2, R, F), 9)
    kept = np.array(M.retained_energy(um, vm, M.row_orthonormalize(gm)))
    np.testing.assert_allclose(kept, np.ones_like(kept), atol=1e-6)


def test_reproject_matches_torch():
    ut, um = _pair((N, TT, K), 10)
    vt, vm = _pair((N, K, F), 11)
    gt, gm = _pair((N, R, F), 12)
    gt, gm = T.row_orthonormalize(gt), M.row_orthonormalize(gm)

    pt = T.reproject_U(ut, vt, gt).numpy()
    pm = np.array(M.reproject_U(um, vm, gm))
    np.testing.assert_allclose(pt, pm, atol=2e-5)


def test_reproject_rebuilds_at_the_STORE_width_not_the_svd_width():
    """Trap 3. r_proj = min(max_rank + oversamples, T_active, feat_dim), so a
    SHORT block makes the SVD narrower than the pool rank. U' has one column per
    BASIS direction, so it does not fit back into the [N, T, r_proj] buffer it
    came from."""
    k_narrow, r_store = 2, 8
    _, um = _pair((1, TT, k_narrow), 13)
    _, vm = _pair((1, k_narrow, F), 14)
    _, gm = _pair((1, r_store, F), 15)
    out = M.reproject_U(um, vm, M.row_orthonormalize(gm))
    assert out.shape == (1, TT, r_store), (
        f"reproject returned {out.shape}, expected the STORE width {r_store} -- "
        f"rebuilding at the SVD width {k_narrow} is trap 3")


def test_projection_is_the_best_approximation_in_the_span():
    """U' Vg must be the ORTHOGONAL projection of U V onto span(Vg): no other
    coefficient matrix in that span gets closer. Perturbing U' can only make the
    residual larger."""
    _, um = _pair((1, TT, K), 16)
    _, vm = _pair((1, K, F), 17)
    _, gm = _pair((1, R, F), 18)
    g = M.row_orthonormalize(gm)

    target = np.array(mx.matmul(um, vm))
    best = np.array(mx.matmul(M.reproject_U(um, vm, g), g))
    base = np.linalg.norm(target - best)

    rng = np.random.default_rng(19)
    for _ in range(5):
        pert = M.reproject_U(um, vm, g) + mx.array(
            rng.standard_normal((1, TT, R)).astype(np.float32) * 0.1)
        alt = np.linalg.norm(target - np.array(mx.matmul(pert, g)))
        assert alt >= base - 1e-4, "found a closer point in the span than the projection"


# ── registry ────────────────────────────────────────────────────────────────

def _reg(capacity, threshold):
    return M.SharedBasisRegistryMLX(capacity=capacity, threshold=threshold,
                                    dtype=mx.float32)


def _store(capacity):
    return mx.zeros((capacity, R, F), dtype=mx.float32)


def test_identical_blocks_share_one_basis():
    _, u1 = _pair((1, TT, R), 20)
    _, v1 = _pair((1, R, F), 21)
    U = mx.concatenate([u1, u1, u1], axis=0)
    V = mx.concatenate([v1, v1, v1], axis=0)

    reg, store = _reg(8, 0.90), _store(8)
    asg, gathered = reg.assign_batch(U, V, 0, store)
    assert len({a.row for a in asg}) == 1, "identical blocks did not share a basis"
    assert reg.stats()["founded"] == 1
    assert reg.stats()["joined"] == 2
    assert reg.stats()["forced"] == 0
    assert gathered.shape == (3, R, F)


def test_orthogonal_blocks_do_not_share():
    """Blocks spanning disjoint subspaces must each found their own group."""
    a = np.zeros((1, R, F), dtype=np.float32); a[0, :, :R] = np.eye(R)
    b = np.zeros((1, R, F), dtype=np.float32); b[0, :, R:2 * R] = np.eye(R)
    _, u = _pair((2, TT, R), 22)
    V = mx.array(np.concatenate([a, b], axis=0))

    reg, store = _reg(8, 0.90), _store(8)
    asg, _ = reg.assign_batch(u, V, 0, store)
    assert asg[0].row != asg[1].row, "orthogonal subspaces were merged"
    assert reg.stats()["founded"] == 2


def test_capacity_is_a_hard_contract_and_forces_joins():
    """Exhausting the store must DEGRADE FIDELITY, never fail a write."""
    rng = np.random.default_rng(23)
    U = mx.array(rng.standard_normal((5, TT, R)).astype(np.float32))
    V = mx.array(rng.standard_normal((5, R, F)).astype(np.float32))

    reg, store = _reg(2, 0.999), _store(2)     # threshold nobody clears
    asg, gathered = reg.assign_batch(U, V, 0, store)

    assert len(asg) == 5, "a write failed instead of degrading"
    assert reg.stats()["forced"] == 3
    assert all(0 <= a.row < 2 for a in asg), "assigned a row outside the store"
    assert not np.isnan(np.array(gathered)).any()


def test_layers_do_not_share_bases():
    _, u1 = _pair((1, TT, R), 24)
    _, v1 = _pair((1, R, F), 25)
    reg, store = _reg(8, 0.90), _store(8)
    a0, _ = reg.assign_batch(u1, v1, 0, store)
    a1, _ = reg.assign_batch(u1, v1, 7, store)
    assert a0[0].row != a1[0].row, (
        "a block found a basis belonging to another layer")


def test_layer_zero_is_not_treated_as_unknown():
    """Trap 6: `getattr(b, 'layer_idx', -1) or -1` makes layer 0 falsy, so every
    layer-0 block reports -1 and shares a search space with unknown blocks.
    Grouping still works -- it groups the WRONG SET, which no aggregate
    statistic reveals."""
    _, u1 = _pair((1, TT, R), 26)
    _, v1 = _pair((1, R, F), 27)
    reg, store = _reg(8, 0.90), _store(8)
    a0, _ = reg.assign_batch(u1, v1, 0, store)
    am, _ = reg.assign_batch(u1, v1, -1, store)
    assert a0[0].row != am[0].row, (
        "layer 0 and layer -1 shared a basis -- layer 0 is being treated as "
        "falsy somewhere")


def test_release_reclaims_a_row_only_when_the_last_member_goes():
    _, u1 = _pair((1, TT, R), 28)
    _, v1 = _pair((1, R, F), 29)
    U = mx.concatenate([u1, u1], axis=0)
    V = mx.concatenate([v1, v1], axis=0)

    reg, store = _reg(4, 0.90), _store(4)
    asg, _ = reg.assign_batch(U, V, 0, store)
    row = asg[0].row
    assert reg.n_groups == 1

    reg.release_row(row)
    assert reg.n_groups == 1, "row reclaimed while a member was still reading it"
    reg.release_row(row)
    assert reg.n_groups == 0
    assert row in reg._free_rows


def test_sharing_factor_is_measured_not_configured():
    """Trap 7: computing it over CAPACITY yields exactly 1/frac whenever the
    store is full, regardless of how many blocks were written."""
    _, u1 = _pair((1, TT, R), 30)
    _, v1 = _pair((1, R, F), 31)
    reg, store = _reg(64, 0.90), _store(64)      # far more capacity than blocks
    U = mx.concatenate([u1] * 4, axis=0)
    V = mx.concatenate([v1] * 4, axis=0)
    reg.assign_batch(U, V, 0, store)
    st = reg.stats()
    assert st["live_rows"] == 1
    assert st["sharing_factor"] == pytest.approx(4.0), (
        f"sharing_factor {st['sharing_factor']} is not the measured "
        f"members-per-live-row")


def test_mean_kept_and_joined_are_reported_together():
    """`joined == 0` with a healthy memory number is the 4-bit-KV signature: the
    saving comes from allocating fewer rows, not from grouping succeeding."""
    reg = _reg(4, 0.90)
    st = reg.stats()
    assert {"joined", "forced", "mean_kept"} <= set(st)
    assert st["mean_kept"] == 1.0, "empty registry should not report a loss"


def test_registry_matches_torch_on_the_same_inputs():
    """End-to-end agreement: same blocks, same threshold, same grouping."""
    rng = np.random.default_rng(32)
    u = rng.standard_normal((5, TT, R)).astype(np.float32)
    v = rng.standard_normal((5, R, F)).astype(np.float32)
    # two exact duplicates so there IS a join to agree about
    u[3], v[3] = u[0], v[0]

    rt = T.SharedBasisRegistry(capacity=8, threshold=0.90, dtype=torch.float32)
    st = torch.zeros((8, R, F), dtype=torch.float32)
    at, _ = rt.assign_batch(torch.from_numpy(u), torch.from_numpy(v), 0, st)

    rm, sm = _reg(8, 0.90), _store(8)
    am, _ = rm.assign_batch(mx.array(u), mx.array(v), 0, sm)

    assert [a.row for a in at] == [a.row for a in am], (
        f"grouping diverged: torch {[a.row for a in at]} vs mlx "
        f"{[a.row for a in am]}")
    assert [a.is_new for a in at] == [a.is_new for a in am]
    assert [a.forced for a in at] == [a.forced for a in am]
    np.testing.assert_allclose([a.kept for a in at], [a.kept for a in am], atol=2e-5)


def test_store_is_mutated_in_place():
    """The store's aliasing semantics are load-bearing and were ASSUMED at first.

    The pool holds `comp_VK`/`comp_VV` and hands a view to the registry; if mx
    `__setitem__` copied instead of mutating, every founded basis would be
    written to a temporary and the decoder would rebuild blocks from a store of
    zeros. Nothing would raise -- the shapes are right and the numbers are
    finite -- so this is pinned rather than trusted.
    """
    _, u1 = _pair((1, TT, R), 33)
    _, v1 = _pair((1, R, F), 34)
    reg, store = _reg(4, 0.90), _store(4)
    assert float(mx.sum(mx.abs(store)).item()) == 0.0

    asg, gathered = reg.assign_batch(u1, v1, 0, store)

    written = np.array(store)[asg[0].row]
    assert np.abs(written).max() > 0.0, (
        "the founded basis is not visible in the caller's store -- mx "
        "__setitem__ is copying, and every block would decompress from zeros")
    np.testing.assert_allclose(written, np.array(gathered)[0], atol=1e-6)

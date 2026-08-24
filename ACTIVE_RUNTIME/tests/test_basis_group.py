"""Shared-basis math: does re-expressing a block in a group basis do what the
docstring claims, and does the registry's capacity contract hold?

These tests are pure math on synthetic factors — no model, no pool, no CUDA
required — so they pin the algorithm independently of the runtime that will
call it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.compression.basis_group import (
    Assignment,
    SharedBasisRegistry,
    reproject_U,
    retained_energy,
    row_orthonormalize,
    shared_basis_enabled,
    shared_basis_fraction,
    shared_basis_threshold,
)


def _rand_factors(N, T, k, F, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(N, T, k, generator=g).to(device)
    V = torch.randn(N, k, F, generator=g).to(device)
    return U, V


# ── row_orthonormalize ───────────────────────────────────────────────────────

def test_orthonormalize_gives_orthonormal_rows():
    _, V = _rand_factors(3, 8, 6, 32, seed=1)
    On = row_orthonormalize(V)
    gram = torch.bmm(On, On.transpose(1, 2))
    eye = torch.eye(6).unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(gram, eye, atol=1e-4), gram[0]


def test_orthonormalize_preserves_span():
    """The orthonormal basis must span exactly what V's rows spanned: projecting
    V onto it has to return V."""
    _, V = _rand_factors(2, 8, 5, 24, seed=2)
    On = row_orthonormalize(V)
    proj = torch.bmm(torch.bmm(V, On.transpose(1, 2)), On)
    assert torch.allclose(proj, V, atol=1e-3), (proj - V).abs().max()


def test_orthonormalize_zeroes_deficient_rows():
    """A rank-deficient V (the shape dynamic_rank truncation leaves) must come
    back with ZERO rows for the missing directions, not arbitrary QR
    complement directions that would let a group claim span it does not have."""
    V = torch.zeros(1, 4, 16)
    V[0, 0] = torch.randn(16)
    V[0, 1] = V[0, 0] * 2.0          # dependent — adds no new direction
    On = row_orthonormalize(V)
    # Exactly one direction is real; the other three rows must be zero.
    nz = (On[0].norm(dim=1) > 1e-5).sum().item()
    assert nz == 1, On[0].norm(dim=1)
    assert torch.isfinite(On).all()


# ── retained_energy ──────────────────────────────────────────────────────────

def test_retained_energy_is_one_for_own_basis():
    """A block projected onto its OWN (orthonormalised) basis loses nothing."""
    U, V = _rand_factors(4, 16, 6, 32, seed=3)
    Vg = row_orthonormalize(V)
    kept = retained_energy(U, V, Vg)                 # [4, 4]
    diag = torch.diagonal(kept)
    assert torch.allclose(diag, torch.ones(4), atol=1e-4), diag


def test_retained_energy_matches_explicit_reconstruction():
    """kept must equal ||U'Vg||^2 / ||UV||^2 computed the slow, obvious way."""
    U, V = _rand_factors(3, 20, 5, 40, seed=4)
    _, Vsrc = _rand_factors(2, 20, 5, 40, seed=5)
    Vg = row_orthonormalize(Vsrc)                     # [2, 5, 40]
    kept = retained_energy(U, V, Vg)                  # [3, 2]

    for n in range(3):
        full = U[n] @ V[n]
        for g in range(2):
            Up = U[n] @ (V[n] @ Vg[g].t())
            approx = Up @ Vg[g]
            expect = (approx.norm() ** 2 / full.norm() ** 2).item()
            assert abs(kept[n, g].item() - expect) < 1e-3, (n, g, kept[n, g].item(), expect)


def test_retained_energy_zero_for_orthogonal_subspace():
    """Disjoint coordinate blocks share nothing."""
    F = 32
    U = torch.randn(1, 10, 4)
    V = torch.zeros(1, 4, F)
    V[0, :, :4] = torch.randn(4, 4)                   # lives in coords 0..3
    Vg = torch.zeros(1, 4, F)
    Vg[0, :, 16:20] = torch.eye(4)                    # lives in coords 16..19
    kept = retained_energy(U, V, Vg)
    assert kept[0, 0].item() < 1e-5, kept


def test_retained_energy_is_one_for_zero_energy_block():
    """A block with no delta energy is reconstructed exactly by any basis —
    reconstructing zero is exact — so it must be reported as a free join, not
    a total loss."""
    U = torch.zeros(1, 8, 4)
    V = torch.randn(1, 4, 16)
    Vg = row_orthonormalize(torch.randn(1, 4, 16))
    kept = retained_energy(U, V, Vg)
    assert kept[0, 0].item() == pytest.approx(1.0)


def test_retained_energy_partial_overlap_is_between():
    """Half the energy in-span, half out, lands near 0.5."""
    F = 32
    V = torch.zeros(1, 2, F)
    V[0, 0, 0] = 1.0          # in the group basis
    V[0, 1, 8] = 1.0          # not in it
    U = torch.zeros(1, 4, 2)
    U[0, :, 0] = 1.0
    U[0, :, 1] = 1.0          # equal energy on both directions
    Vg = torch.zeros(1, 2, F)
    Vg[0, 0, 0] = 1.0
    kept = retained_energy(U, V, Vg)
    assert 0.45 < kept[0, 0].item() < 0.55, kept


# ── reproject_U ──────────────────────────────────────────────────────────────

def test_reproject_reconstructs_exactly_in_own_basis():
    """U' Vg must reproduce U V bit-for-bit-ish when Vg spans V."""
    U, V = _rand_factors(3, 24, 6, 48, seed=6)
    Vg = row_orthonormalize(V)
    Up = reproject_U(U, V, Vg)
    recon = torch.bmm(Up, Vg)
    orig = torch.bmm(U.float(), V.float())
    assert torch.allclose(recon, orig, atol=1e-3), (recon - orig).abs().max()


def test_reproject_is_the_optimal_approximation():
    """No other coefficient matrix beats U (V Vg^T) — verify against an explicit
    least-squares solve."""
    U, V = _rand_factors(1, 30, 5, 40, seed=7)
    Vg = row_orthonormalize(torch.randn(1, 5, 40))
    Up = reproject_U(U, V, Vg)
    target = (U[0] @ V[0])                                  # [T, F]
    lstsq = torch.linalg.lstsq(Vg[0].t(), target.t()).solution.t()   # [T, r]
    assert torch.allclose(Up[0], lstsq, atol=1e-3), (Up[0] - lstsq).abs().max()


def test_reproject_error_matches_retained_energy():
    """The energy retained_energy predicts is the energy reproject actually keeps."""
    U, V = _rand_factors(2, 20, 5, 36, seed=8)
    Vg = row_orthonormalize(torch.randn(2, 5, 36))
    kept = retained_energy(U, V, Vg)                        # [2, 2]
    for n in range(2):
        Up = reproject_U(U[n:n+1], V[n:n+1], Vg[n:n+1])
        recon = torch.bmm(Up, Vg[n:n+1])[0]
        orig = U[n].float() @ V[n].float()
        actual = (recon.norm() ** 2 / orig.norm() ** 2).item()
        assert abs(actual - kept[n, n].item()) < 1e-3, (n, actual, kept[n, n].item())


# ── SharedBasisRegistry ──────────────────────────────────────────────────────

def _store(capacity, r, F, device="cpu", dtype=torch.float32):
    return torch.zeros((capacity, r, F), device=device, dtype=dtype)


def test_registry_identical_blocks_share_one_row():
    """Blocks spanning the SAME subspace must collapse onto one basis row —
    this is the whole point of the feature."""
    U, V = _rand_factors(1, 16, 4, 32, seed=9)
    U = U.repeat(5, 1, 1)
    V = V.repeat(5, 1, 1)
    reg = SharedBasisRegistry(capacity=8, threshold=0.9)
    store = _store(8, 4, 32)
    asg, gathered = reg.assign_batch(U, V, layer=0, basis_store=store)
    assert reg.n_groups == 1, reg.stats()
    assert len({a.row for a in asg}) == 1
    assert asg[0].is_new and not any(a.is_new for a in asg[1:])
    assert reg.n_forced == 0
    assert gathered.shape == (5, 4, 32)


def test_registry_unrelated_blocks_get_own_rows():
    """Blocks in disjoint subspaces must NOT be merged."""
    F = 64
    N = 4
    U = torch.randn(N, 12, 4)
    V = torch.zeros(N, 4, F)
    for n in range(N):
        V[n, :, n * 8:(n * 8) + 4] = torch.eye(4)     # four disjoint coord blocks
    reg = SharedBasisRegistry(capacity=8, threshold=0.9)
    store = _store(8, 4, F)
    asg, _ = reg.assign_batch(U, V, layer=0, basis_store=store)
    assert reg.n_groups == N, reg.stats()
    assert len({a.row for a in asg}) == N
    assert all(a.is_new for a in asg)


def test_registry_capacity_forces_a_join_rather_than_failing():
    """When the store is full a block MUST still get a row — capacity degrades
    fidelity, never correctness."""
    F = 64
    N = 5
    U = torch.randn(N, 12, 4)
    V = torch.zeros(N, 4, F)
    for n in range(N):
        V[n, :, n * 8:(n * 8) + 4] = torch.eye(4)     # all mutually orthogonal
    reg = SharedBasisRegistry(capacity=2, threshold=0.9)
    store = _store(2, 4, F)
    asg, gathered = reg.assign_batch(U, V, layer=0, basis_store=store)
    assert len(asg) == N
    assert reg.n_groups == 2
    assert reg.n_forced == 3, reg.stats()
    assert all(0 <= a.row < 2 for a in asg)
    # The forced ones are honest about what they lost.
    assert all(a.kept < 0.9 for a in asg if a.forced)
    assert torch.isfinite(gathered).all()


def test_registry_threshold_controls_merging():
    """A low threshold merges what a high threshold keeps apart."""
    F = 32
    V = torch.zeros(2, 2, F)
    V[0, 0, 0] = 1.0
    V[0, 1, 1] = 1.0
    V[1, 0, 0] = 1.0
    V[1, 1, 8] = 1.0            # 1 of 2 directions shared
    U = torch.ones(2, 6, 2)

    strict = SharedBasisRegistry(capacity=4, threshold=0.9)
    a_strict, _ = strict.assign_batch(U, V, layer=0, basis_store=_store(4, 2, F))
    assert strict.n_groups == 2, strict.stats()

    loose = SharedBasisRegistry(capacity=4, threshold=0.4)
    a_loose, _ = loose.assign_batch(U, V, layer=0, basis_store=_store(4, 2, F))
    assert loose.n_groups == 1, loose.stats()
    assert a_loose[1].kept == pytest.approx(0.5, abs=0.05)


def test_registry_layers_are_isolated():
    """The same subspace in a different layer must not be handed a foreign basis
    while free rows remain."""
    U, V = _rand_factors(1, 16, 4, 32, seed=10)
    reg = SharedBasisRegistry(capacity=8, threshold=0.9)
    store = _store(8, 4, 32)
    reg.assign_batch(U, V, layer=0, basis_store=store)
    reg.assign_batch(U, V, layer=1, basis_store=store)
    assert reg.n_groups == 2, reg.stats()


def test_registry_release_reclaims_the_row():
    """Freeing a group's last member returns its row to the free list, so a long
    session that cycles topics does not permanently lose basis capacity."""
    U, V = _rand_factors(1, 16, 4, 32, seed=11)
    reg = SharedBasisRegistry(capacity=2, threshold=0.9)
    store = _store(2, 4, 32)
    asg, _ = reg.assign_batch(U.repeat(3, 1, 1), V.repeat(3, 1, 1), layer=0,
                              basis_store=store)
    row = asg[0].row
    assert reg.n_groups == 1
    reg.release_row(row)
    reg.release_row(row)
    assert reg.n_groups == 1, "two of three members released — group still live"
    reg.release_row(row)
    assert reg.n_groups == 0, reg.stats()
    # And the row is reusable.
    asg2, _ = reg.assign_batch(U, V, layer=5, basis_store=store)
    assert asg2[0].is_new and reg.n_groups == 1


def test_registry_reprojection_is_lossless_when_shared_legitimately():
    """End-to-end: blocks that legitimately share a subspace must reconstruct
    through the SHARED basis as well as through their own."""
    torch.manual_seed(12)
    F, k = 48, 5
    common = row_orthonormalize(torch.randn(1, k, F))[0]        # [k, F]
    N = 6
    U = torch.randn(N, 20, k)
    # Every block's V is a different mixing of the SAME k directions.
    V = torch.stack([torch.randn(k, k) @ common for _ in range(N)])

    reg = SharedBasisRegistry(capacity=8, threshold=0.9)
    store = _store(8, k, F)
    asg, gathered = reg.assign_batch(U, V, layer=0, basis_store=store)
    assert reg.n_groups == 1, reg.stats()

    Up = reproject_U(U, V, gathered)
    recon = torch.bmm(Up, gathered)
    orig = torch.bmm(U.float(), V.float())
    rel = ((recon - orig).norm() / orig.norm()).item()
    assert rel < 1e-3, rel


def test_registry_stats_are_consistent():
    U, V = _rand_factors(6, 12, 4, 32, seed=13)
    reg = SharedBasisRegistry(capacity=3, threshold=0.95)
    asg, _ = reg.assign_batch(U, V, layer=0, basis_store=_store(3, 4, 32))
    s = reg.stats()
    assert s["founded"] + s["joined"] + s["forced"] == len(asg) == 6
    assert s["groups"] <= s["capacity"] == 3
    assert 0.0 <= s["mean_kept"] <= 1.0


def test_registry_prescreen_does_not_change_a_clear_winner(monkeypatch):
    """With a narrow prescreen the exact winner must still be found when it is
    the obvious top-direction match."""
    F, k = 64, 3
    N = 12
    torch.manual_seed(14)
    V = torch.zeros(N, k, F)
    for n in range(N):
        base = (n % 4) * 12
        V[n, :, base:base + k] = torch.eye(k)
    U = torch.randn(N, 10, k)

    monkeypatch.setenv("DKV_SHARED_BASIS_PRESCREEN", "2")
    reg = SharedBasisRegistry(capacity=16, threshold=0.9)
    asg, _ = reg.assign_batch(U, V, layer=0, basis_store=_store(16, k, F))
    assert reg.n_groups == 4, reg.stats()
    assert reg.n_forced == 0


# ── config knobs ─────────────────────────────────────────────────────────────

def test_feature_is_off_by_default(monkeypatch):
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    assert shared_basis_enabled() is False


@pytest.mark.parametrize("val,expect", [("1", True), ("on", True), ("0", False),
                                        ("off", False), ("", False)])
def test_enable_flag_parsing(monkeypatch, val, expect):
    monkeypatch.setenv("DKV_SHARED_BASIS", val)
    assert shared_basis_enabled() is expect


def test_fraction_and_threshold_are_clamped(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "5.0")
    assert shared_basis_fraction() == 1.0
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "-1")
    assert shared_basis_fraction() > 0.0
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "garbage")
    assert shared_basis_fraction() == 0.25
    monkeypatch.setenv("DKV_SHARED_BASIS_THRESHOLD", "2.0")
    assert shared_basis_threshold() == 1.0
    monkeypatch.setenv("DKV_SHARED_BASIS_THRESHOLD", "nope")
    assert shared_basis_threshold() == 0.90


# ── The two projection defects the MLX port handed back (CUDA_TODO §2b) ──────
#
# Both were invisible to every test that existed: reconstruction stayed exact
# under the old code, so no distance metric moved. What moved was the split of
# scale BETWEEN U and V, which only the router can see.


def _nonorthonormal_basis(r, F, seed=0):
    """A basis shaped like the one this pool actually stores.

    The joint ``[K | V]`` basis is sliced out of one orthonormal ``Vh`` and its
    V half is then divided by the per-block ``v_scale`` gain, so the rows are
    not unit-norm and not mutually orthogonal. Measured row norms on the MLX
    port: 0.78-0.83.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    On = row_orthonormalize(torch.randn(1, r, F, generator=g))[0]
    half = F // 2
    V = On.clone()
    V[:, half:] /= 3.0                       # the v_scale gain, divided back out
    return V


def test_founder_reprojection_is_exact_with_its_own_basis():
    """U' == U when the group basis IS the block's own V.

    This is the whole point of the pseudo-inverse. ``U (V Vg^T)`` gets this
    wrong for a non-orthonormal Vg -- and every block founds its own group when
    nothing shares, so it is the common case, not an edge one.
    """
    F, r, T = 64, 8, 16
    V = _nonorthonormal_basis(r, F, seed=3).unsqueeze(0)          # [1, r, F]
    g = torch.Generator(device="cpu").manual_seed(4)
    U = torch.randn(1, T, r, generator=g)

    Up = reproject_U(U, V, V)
    assert torch.allclose(Up, U, atol=1e-4), (
        f"founder is not exact: max|U' - U| = {(Up - U).abs().max():.3e}")
    # and therefore the reconstruction is unchanged too
    assert torch.allclose(torch.bmm(Up, V), torch.bmm(U, V), atol=1e-3)


def test_reproject_recovers_the_true_projection_for_a_skewed_basis():
    """U' Vg must be the ORTHOGONAL projection of U V onto span(Vg).

    Checked by residual orthogonality: (U V - U' Vg) must be perpendicular to
    every row of Vg. The transpose form fails this whenever Vg Vg^T != I.
    """
    F, r, k, T = 48, 6, 6, 12
    Vg = _nonorthonormal_basis(r, F, seed=7).unsqueeze(0)
    g = torch.Generator(device="cpu").manual_seed(8)
    U = torch.randn(1, T, k, generator=g)
    V = torch.randn(1, k, F, generator=g)

    resid = torch.bmm(U, V) - torch.bmm(reproject_U(U, V, Vg), Vg)   # [1, T, F]
    leak = torch.matmul(resid, Vg.transpose(1, 2)).abs().max()       # [1, T, r]
    assert leak < 1e-3, f"residual is not orthogonal to the basis: {leak:.3e}"


def test_retained_energy_is_one_for_a_block_in_its_own_skewed_basis():
    """A block scored against its own basis keeps ALL of its energy.

    With the plain ``C C^T`` form and a basis whose rows are scaled by 1/3, the
    score is off by that scaling and the threshold decision changes -- so a
    block can be refused entry to the one group that represents it exactly.
    """
    F, r, T = 48, 6, 20
    V = _nonorthonormal_basis(r, F, seed=11).unsqueeze(0)          # [1, r, F]
    g = torch.Generator(device="cpu").manual_seed(12)
    U = torch.randn(1, T, r, generator=g)

    kept = retained_energy(U, V, V)                                # [1, 1]
    assert kept.shape == (1, 1)
    assert abs(float(kept[0, 0]) - 1.0) < 1e-3, (
        f"self-energy should be 1.0, got {float(kept[0, 0]):.4f}")


def test_retained_energy_still_matches_the_orthonormal_form():
    """The solve must REDUCE to the old formula when Vg really is orthonormal.

    Guards the reduction claim in the comment: if this drifts, every previously
    measured shared-basis number becomes incomparable.
    """
    F, r, k, T, G = 64, 8, 8, 24, 3
    g = torch.Generator(device="cpu").manual_seed(21)
    U = torch.randn(2, T, k, generator=g)
    V = torch.randn(2, k, F, generator=g)
    Vg = row_orthonormalize(torch.randn(G, r, F, generator=g))

    kept = retained_energy(U, V, Vg)
    # old form, computed here directly
    Gram = torch.bmm(U.transpose(1, 2), U)
    den = (Gram * torch.bmm(V, V.transpose(1, 2))).sum(dim=(1, 2))
    C = torch.matmul(V, Vg.reshape(G * r, F).t()).reshape(2, k, G, r).permute(0, 2, 1, 3)
    num = (Gram.unsqueeze(1) * torch.matmul(C, C.transpose(-1, -2))).sum(dim=(-1, -2))
    ref = (num / den.clamp(min=1e-12).unsqueeze(1)).clamp(0.0, 1.0)
    assert torch.allclose(kept, ref, atol=1e-4), (
        f"solve form diverged from the orthonormal form: "
        f"max {(kept - ref).abs().max():.3e}")


def test_founder_stores_an_orthonormal_basis_and_reprojection_survives_it():
    """CUDA stores the ORTHONORMALISED founding basis, and that is deliberate.

    The MLX port stores the raw V so a founder reprojects to exactly its own U.
    CUDA cannot: this pool quantises U to int8 with one per-block scale, and a
    raw joint [K|V] basis is ill-conditioned enough that U' = U V Vg^+ carries
    that conditioning into the quantised tensor. Measured on
    test_shared_blocks_still_reconstruct, six blocks on one basis --
    orthonormal store: joiner rel 0.0058-0.0079; raw store: 0.0379-0.0791.

    What must hold either way is that RECONSTRUCTION is unchanged: U' Vg is the
    same projection of U V regardless of which basis of the same row space is
    stored. That is the invariant this pins, so a future change of store cannot
    silently move it.
    """
    F, r, T = 32, 4, 10
    V = _nonorthonormal_basis(r, F, seed=31).unsqueeze(0)          # [1, r, F]
    g = torch.Generator(device="cpu").manual_seed(32)
    U = torch.randn(1, T, r, generator=g)

    reg = SharedBasisRegistry(capacity=4, threshold=0.9)
    store = torch.zeros(4, r, F)
    assignments, gathered = reg.assign_batch(U, V, layer=0, basis_store=store)

    assert assignments[0].is_new, "first block must found its own group"
    row = assignments[0].row
    norms = store[row].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        f"founder row is not orthonormal: norms {norms.tolist()}")

    # The invariant that matters: same row space -> same reconstruction.
    rec_store = torch.bmm(reproject_U(U, V, gathered), gathered)
    rec_own = torch.bmm(reproject_U(U, V, V), V)
    assert torch.allclose(rec_store, rec_own, atol=1e-3), (
        "changing which basis of the row space is stored moved the "
        "reconstruction; reproject_U is not taking a true projection")

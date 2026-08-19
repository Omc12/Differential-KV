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

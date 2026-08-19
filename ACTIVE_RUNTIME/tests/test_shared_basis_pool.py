"""Shared-basis storage in NativeBlockPool.

The pool is the load-bearing part of the change: V_KV stops being one row per
slot and becomes a basis store reached through `basis_of`. These tests cover
the two things that matter --

  1. With DKV_SHARED_BASIS off (the default, and what the rest of the suite
     runs under) the pool allocates and reconstructs EXACTLY as before.
  2. With it on, blocks still reconstruct to the right K/V, VRAM actually
     drops, and the capacity/refcount contracts hold.

Reconstruction is checked end to end -- write a block, read it back through
the pool's own indirection, compare against anchor + U V -- because that is
the only assertion that catches an indirection wired up in one reader and not
another.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime.native_block_pool import NativeBlockPool, _JointVAdapter


KV, HD, RANK, SEQ = 2, 16, 8, 12
FEAT = 2 * KV * HD


def _pool(n_blocks=16, **kw):
    p = NativeBlockPool(
        max_blocks=1024, num_kv_heads=KV, head_dim=HD, rank=RANK,
        max_seq_len=SEQ, device="cpu", dtype=torch.float32,
        initial_blocks=n_blocks, num_layers=2, lazy=False,
        max_residual_tokens=4, **kw)
    return p


def _factors(n, k=RANK, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(n, SEQ, k, generator=g)
    V = torch.randn(n, k, FEAT, generator=g)
    return U, V


def _shared_factors(n, k=RANK, seed=0):
    """n blocks whose V rows all live in ONE k-dimensional subspace."""
    g = torch.Generator().manual_seed(seed)
    common = torch.linalg.qr(torch.randn(FEAT, k, generator=g)).Q.t()   # [k, FEAT]
    U = torch.randn(n, SEQ, k, generator=g)
    V = torch.stack([torch.randn(k, k, generator=g) @ common for _ in range(n)])
    return U, V


def _write(pool, slots, U, V, layer=0):
    n = len(slots)
    pool.write_blocks_batched(
        pool_indices=torch.tensor(slots, dtype=torch.long),
        U=U, V=V,
        anchor_K=torch.zeros(n, KV, HD),
        anchor_V=torch.zeros(n, KV, HD),
        scales=torch.ones(n),
        seq_len=SEQ,
        layer_idx=layer,
    )


def _readback_delta_K(pool, slot):
    """Reconstruct one block's K-side delta the way the decode path does:
    U (int8 * scale) @ V_K, with V looked up through the basis map."""
    r = pool.basis_row(slot)
    U = pool.U[slot].float() * pool.U_scale[slot].float()      # [SEQ, RANK]
    VK = pool.V_KV[r, 0].float().reshape(RANK, KV * HD)        # [RANK, KV*HD]
    return U @ VK


# ── default path is untouched ────────────────────────────────────────────────

def test_off_by_default_allocates_one_v_row_per_slot(monkeypatch):
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    p = _pool(16)
    assert p.shared_basis_active is False
    assert p.basis_of is None
    assert p.V_KV.shape[0] == p.current_blocks == 16
    assert p.basis_index(torch.tensor([3, 7])).tolist() == [3, 7]
    assert p.basis_row(5) == 5
    assert p.basis_stats() == {"enabled": False}


def test_off_by_default_reconstruction_is_the_old_behaviour(monkeypatch):
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    p = _pool(8)
    U, V = _factors(3, seed=1)
    _write(p, [0, 1, 2], U, V)
    for i in range(3):
        got = _readback_delta_K(p, i)
        want = U[i].float() @ V[i, :, :KV * HD].float()
        # int8 U quantisation is the only loss here.
        rel = ((got - want).norm() / want.norm()).item()
        assert rel < 0.02, (i, rel)


def test_release_basis_is_a_noop_when_off(monkeypatch):
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    p = _pool(8)
    p.release_basis(3)          # must not raise
    p.free_block(0)


# ── shared basis on ──────────────────────────────────────────────────────────

def test_store_is_smaller_than_the_block_count(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.25")
    p = _pool(16)
    assert p.shared_basis_active is True
    assert p.V_KV.shape[0] == 4, p.V_KV.shape
    assert p.basis_of.shape[0] == 16


def test_bytes_per_block_amortises_v(monkeypatch):
    """The saving has to reach the SIZING arithmetic, or the pool just holds
    the same number of blocks in less memory instead of more blocks."""
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    plain = _pool(8)
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.25")
    shared = _pool(8)

    v_bytes = RANK * KV * HD * 2 * 2
    assert shared._bytes_per_block < plain._bytes_per_block
    saved = plain._bytes_per_block - shared._bytes_per_block
    assert saved == pytest.approx(v_bytes * 0.75, rel=0.01), (saved, v_bytes)


def test_pool_vram_actually_drops(monkeypatch):
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    plain_mb = _pool(64)._pool_mb()
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.25")
    shared_mb = _pool(64)._pool_mb()
    assert shared_mb < plain_mb, (plain_mb, shared_mb)


def test_blocks_in_one_subspace_collapse_to_one_row(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(16)
    U, V = _shared_factors(6, seed=2)
    _write(p, list(range(6)), U, V)
    rows = {p.basis_row(i) for i in range(6)}
    assert len(rows) == 1, (rows, p.basis_stats())
    assert p.basis_stats()["groups"] == 1


def test_shared_blocks_still_reconstruct(monkeypatch):
    """The point of the whole feature: sharing a basis must not change what a
    block decompresses to, when the blocks genuinely share a subspace."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(16)
    U, V = _shared_factors(6, seed=3)
    _write(p, list(range(6)), U, V)
    for i in range(6):
        got = _readback_delta_K(p, i)
        want = U[i].float() @ V[i, :, :KV * HD].float()
        rel = ((got - want).norm() / want.norm()).item()
        assert rel < 0.05, (i, rel, p.basis_stats())


def test_unrelated_blocks_get_their_own_rows(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "1.0")
    p = _pool(8)
    U, V = _factors(4, seed=4)
    _write(p, [0, 1, 2, 3], U, V)
    rows = {p.basis_row(i) for i in range(4)}
    assert len(rows) == 4, (rows, p.basis_stats())


def test_capacity_pressure_forces_joins_without_failing(monkeypatch):
    """A frac too small for the document must degrade fidelity, never error."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.125")   # 16 slots -> 2 rows
    p = _pool(16)
    U, V = _factors(8, seed=5)
    _write(p, list(range(8)), U, V)
    st = p.basis_stats()
    assert st["groups"] <= 2
    assert st["forced"] > 0, st
    assert all(0 <= p.basis_row(i) < 2 for i in range(8))
    for i in range(8):
        assert torch.isfinite(_readback_delta_K(p, i)).all()


def test_layers_do_not_share_rows(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "1.0")
    p = _pool(8)
    U, V = _shared_factors(2, seed=6)
    _write(p, [0, 1], U, V, layer=0)
    _write(p, [2, 3], U, V, layer=1)
    assert p.basis_row(0) == p.basis_row(1)
    assert p.basis_row(2) == p.basis_row(3)
    assert p.basis_row(0) != p.basis_row(2), p.basis_stats()


def test_freeing_a_slot_reclaims_its_basis_row(monkeypatch):
    """Without this a session that cycles topics exhausts basis capacity and
    every later block is force-joined -- a fidelity cliff with no error."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.25")   # 16 slots -> 4 rows
    p = _pool(16)
    U, V = _factors(4, seed=7)
    slots = p.allocate_blocks(4)
    _write(p, slots, U, V)
    assert p.basis_stats()["groups"] == 4
    for s in slots:
        p.free_block(s)
    assert p.basis_stats()["groups"] == 0, p.basis_stats()

    # And the rows are reusable by a completely different set of blocks.
    U2, V2 = _factors(4, seed=8)
    slots2 = p.allocate_blocks(4)
    _write(p, slots2, U2, V2)
    assert p.basis_stats()["groups"] == 4
    assert p.basis_stats()["forced"] == 0


def test_rewriting_a_slot_does_not_leak_its_old_claim(monkeypatch):
    """Overwriting a slot in place must release the previous basis claim, or
    the refcount only ever rises."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(8)
    U, V = _factors(1, seed=9)
    for _ in range(5):
        _write(p, [0], U, V)
    assert p.basis_stats()["groups"] == 1, p.basis_stats()
    p.free_block(0)
    assert p.basis_stats()["groups"] == 0, p.basis_stats()


def test_growth_preserves_existing_assignments(monkeypatch):
    """Growth only APPENDS basis rows, so an already-written slot keeps its row
    and never has to be re-projected."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(8)
    U, V = _factors(3, seed=10)
    _write(p, [0, 1, 2], U, V)
    before = [p.basis_row(i) for i in range(3)]
    recon_before = [_readback_delta_K(p, i).clone() for i in range(3)]

    p._grow_increment = 8
    p._grow_pool()
    assert p.current_blocks > 8
    assert p.basis_of.shape[0] == p.current_blocks
    assert p.V_KV.shape[0] == p._n_basis_rows(p.current_blocks)
    assert [p.basis_row(i) for i in range(3)] == before
    for i in range(3):
        assert torch.allclose(_readback_delta_K(p, i), recon_before[i], atol=1e-5)


def test_growth_hands_the_registry_its_new_rows(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(8)
    cap_before = p.basis_registry.capacity
    p._grow_increment = 8
    p._grow_pool()
    assert p.basis_registry.capacity == p._n_basis_rows(p.current_blocks)
    assert p.basis_registry.capacity > cap_before
    assert len(p.basis_registry._free_rows) > 0


def test_correction_form_residuals_refuse_to_share(monkeypatch):
    """Residuals in CORRECTION form are defined against a block's OWN low-rank
    reconstruction, so re-expressing it in a shared basis would invalidate every
    one of them. The pool must refuse rather than silently corrupt."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_RESIDUAL_EXACT_KEYS", "0")
    p = _pool(8)
    assert p.shared_basis_active is False
    assert p.V_KV.shape[0] == p.current_blocks


def test_per_block_write_path_shares_too(monkeypatch):
    """write_block (CPU compress / fallback) must honour the same contract as
    write_blocks_batched, or the two paths disagree about where V lives."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(8)
    U, V = _shared_factors(3, seed=11)
    for i in range(3):
        p.write_block(pool_idx=i, U=U[i], V=V[i],
                      anchor_K=torch.zeros(KV, HD), anchor_V=torch.zeros(KV, HD),
                      scale=1.0, seq_len=SEQ, layer_idx=0)
    assert len({p.basis_row(i) for i in range(3)}) == 1, p.basis_stats()
    for i in range(3):
        got = _readback_delta_K(p, i)
        want = U[i].float() @ V[i, :, :KV * HD].float()
        assert ((got - want).norm() / want.norm()).item() < 0.05


def test_unwritten_slot_resolves_to_a_valid_row(monkeypatch):
    """An unwritten slot must still index somewhere in range -- any gather over
    it would otherwise read out of bounds."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.25")
    p = _pool(16)
    idx = torch.arange(16)
    rows = p.basis_index(idx)
    assert rows.min().item() >= 0
    assert rows.max().item() < p.V_KV.shape[0]
    # ...and it reconstructs to the anchor, because its U is zero.
    assert _readback_delta_K(p, 9).abs().max().item() == 0.0


def test_basis_index_clamps_out_of_range_slots(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    p = _pool(8)
    rows = p.basis_index(torch.tensor([-5, 999]))
    assert rows.min().item() >= 0 and rows.max().item() < p.V_KV.shape[0]
    assert p.basis_row(-5) == 0 and p.basis_row(999) == 0


def test_basis_stats_reports_the_sharing_factor(monkeypatch):
    """Sharing factor must count slots that actually HOLD a claim.

    Over pool capacity it would report 1/frac whenever the store is full no
    matter how many blocks were written -- i.e. it would report the config back
    as if it were a measurement. Here 8 of 16 slots are written onto 1 group,
    so the honest answer is 8x, not 16x.
    """
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_SHARED_BASIS_FRAC", "0.5")
    p = _pool(16)
    U, V = _shared_factors(8, seed=12)
    _write(p, list(range(8)), U, V)
    st = p.basis_stats()
    assert st["enabled"] is True
    assert st["groups"] == 1
    assert st["slots"] == 16
    assert st["claimed"] == 8
    assert st["sharing_factor"] == pytest.approx(8.0)


# ── regressions found on real hardware, not by the CPU tests above ───────────

def test_compiled_kernels_decline_under_shared_basis(monkeypatch):
    """The dkv_core / Metal decode kernels take the WHOLE pool.V_K and index it
    by SLOT internally. Under shared bases that is wrong twice: slots share
    rows, and the store is SHORTER than the slot count, so a high slot id reads
    out of bounds -- a CUDA device-side assert, not a Python error. Neither can
    be redirected from Python, so those paths must decline.

    This is the bug the CPU pool tests could not see: they drive the pool
    directly and never reach a kernel dispatch.
    """
    from runtime.dkv_attention import _compiled_kernel_ok

    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    assert _compiled_kernel_ok(_pool(8)) is True

    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    assert _compiled_kernel_ok(_pool(8)) is False
    # ...and anything that is not a pool at all must not crash the guard.
    assert _compiled_kernel_ok(None) is True


def test_every_compiled_kernel_dispatch_is_guarded():
    """All five dispatch sites, not just the ones that happen to run here --
    two of them are macOS-only and one needs an opt-in native build, so a
    missing guard would stay invisible on this machine indefinitely."""
    import runtime.dkv_attention as da
    src = open(da.__file__, encoding="utf-8").read()
    for marker in ("_dkv_core, \"fused_decode_attention_combined\"",
                   "_DKV_HAS_METAL_ATTN and pool is not None",
                   "_DKV_HAS_DECODE_ATTN and pool is not None"):
        for line in src.splitlines():
            if marker in line and line.strip().startswith(("if ", "elif ")):
                assert "_compiled_kernel_ok(pool)" in line, line.strip()


def test_reprojection_widens_U_to_the_store_rank():
    """r_proj is min(max_rank + oversamples, T_active, feat_dim), so a short
    block makes the SVD NARROWER than the pool rank. U' has one column per
    BASIS direction, so it cannot be written back into the [N, T, r_proj]
    buffer it came from -- the compress path must rebuild at the store width.
    """
    from native_core.compression.basis_group import reproject_U, row_orthonormalize
    N, T, r_proj, r_store, F = 2, 10, 5, 12, 32
    U = torch.randn(N, T, r_proj)
    V = torch.randn(N, r_proj, F)
    Vg = row_orthonormalize(torch.randn(N, r_store, F))
    Up = reproject_U(U, V, Vg)
    assert Up.shape == (N, T, r_store), Up.shape
    # ...and it still reconstructs: r_store >= r_proj means the group basis can
    # represent everything the block's own basis could, if it spans it.
    Vg_full = row_orthonormalize(
        torch.cat([V, torch.zeros(N, r_store - r_proj, F)], dim=1))
    recon = torch.bmm(reproject_U(U, V, Vg_full), Vg_full)
    assert torch.allclose(recon, torch.bmm(U, V), atol=1e-3)


def test_layer_zero_is_not_reported_as_unknown():
    """`getattr(b, 'layer_idx', -1) or -1` makes layer 0 falsy, so every
    layer-0 block reported -1 and searched the same basis space as genuinely
    unknown blocks. Grouping still worked -- it grouped the wrong set."""
    from native_core.compression.lowrank import _batch_layer_idx

    class _B:
        def __init__(self, li):
            self.layer_idx = li

    assert _batch_layer_idx([_B(0)]) == 0
    assert _batch_layer_idx([_B(7)]) == 7
    assert _batch_layer_idx([_B(None)]) == -1
    assert _batch_layer_idx([]) == -1


# ── the V-layout adapter ─────────────────────────────────────────────────────

def test_joint_adapter_round_trips():
    """The adapter is the only thing that knows the pool's split K/V layout and
    the registry's joint [r, F] layout are the same bytes. If it transposes
    them wrongly, every shared block decompresses to garbage."""
    V_KV = torch.zeros(3, 2, RANK, KV, HD)
    a = _JointVAdapter(V_KV)
    assert a.shape == (3, RANK, FEAT)

    joint = torch.randn(RANK, FEAT)
    a[1] = joint
    assert torch.allclose(a[1], joint, atol=1e-6)
    # ...and it landed in the split layout the kernels read.
    half = FEAT // 2
    assert torch.allclose(V_KV[1, 0], joint[:, :half].reshape(RANK, KV, HD), atol=1e-6)
    assert torch.allclose(V_KV[1, 1], joint[:, half:].reshape(RANK, KV, HD), atol=1e-6)


def test_joint_adapter_batched_gather():
    V_KV = torch.zeros(4, 2, RANK, KV, HD)
    a = _JointVAdapter(V_KV)
    j0, j2 = torch.randn(RANK, FEAT), torch.randn(RANK, FEAT)
    a[0] = j0
    a[2] = j2
    got = a[torch.tensor([0, 2])]
    assert got.shape == (2, RANK, FEAT)
    assert torch.allclose(got[0], j0, atol=1e-6)
    assert torch.allclose(got[1], j2, atol=1e-6)

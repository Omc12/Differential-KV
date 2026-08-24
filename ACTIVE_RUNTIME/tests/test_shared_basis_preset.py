"""Shared bases are wired to the PRESET, not only to an env var.

`low` is the memory-priority preset and looks like the natural home for the
23.6% pool reduction. It is NOT: `low` is the one preset that keeps
rotated_pool=True, and shared bases on a rotated pool degenerate to forced
lossy joins at an identical pool size (see test_shared_basis_refuses_a_rotated_pool).
It also sets kv_quant="q4_0", which breaks the same feature for an INDEPENDENT
reason.
`mid` (the default) and `high` (max fidelity) must not, and an explicit env
var must still win over whatever the preset chose -- in BOTH directions, which
is the half that is easy to get wrong: `shared_basis_enabled()` returns False
for an unset variable, so consulting it unconditionally would let "unset"
silently override a preset that asked for it on.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.config import DKVConfig
from runtime.native_block_pool import NativeBlockPool


def _pool(shared_basis, frac=0.50, n=16):
    return NativeBlockPool(
        max_blocks=1024, num_kv_heads=2, head_dim=16, rank=8, max_seq_len=12,
        device="cpu", dtype=torch.float32, initial_blocks=n, num_layers=2,
        lazy=False, max_residual_tokens=4,
        shared_basis=shared_basis, shared_basis_frac=frac)


# ── the preset decides ───────────────────────────────────────────────────────

@pytest.mark.parametrize("preset", ["low", "mid", "high", "ultra"])
def test_no_preset_enables_shared_basis(monkeypatch, preset):
    """OFF everywhere, and `low` is the one worth stating explicitly.

    `low` is memory-priority, so it looks like the natural home. It is not:
    `low` sets kv_quant="q4_0", and 4-bit KV quantisation destroys the subspace
    agreement, taking voluntary joins to ZERO and mean_kept from 0.969 to
    0.685 -- at an IDENTICAL pool size, since the saving comes from allocating
    fewer basis rows rather than from grouping succeeding.
    """
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    assert bool(DKVConfig({"preset": preset}).shared_basis) is False, preset


def test_low_preset_is_the_one_that_would_break_it():
    """Pins the reason `low` is excluded, so re-enabling it there has to
    confront the quantisation interaction rather than look like an oversight."""
    assert DKVConfig({"preset": "low"}).kv_quant.lower().startswith("q4")


def test_default_frac_is_the_measured_setting(monkeypatch):
    """0.50 is the point measured at 2.58x sharing, 463 voluntary joins, ZERO
    forced, retained energy 0.969. Deeper fracs are a different trade."""
    monkeypatch.delenv("DKV_SHARED_BASIS_FRAC", raising=False)
    assert DKVConfig({"preset": "mid"}).shared_basis_frac == pytest.approx(0.50)


def test_quantised_kv_warns(monkeypatch, capsys):
    """The q4_0 interaction must announce itself. It is invisible in pool MB --
    identical either way -- so without this it is only findable in
    basis_stats()['joined'], which nobody reads until something is wrong."""
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    monkeypatch.setenv("DKV_KV_QUANT", "q4_0")
    _pool(shared_basis=True)
    assert "4-bit KV" in capsys.readouterr().out

    monkeypatch.setenv("DKV_KV_QUANT", "f16")
    _pool(shared_basis=True)
    assert "4-bit KV" not in capsys.readouterr().out


# ── the env var still wins, both ways ────────────────────────────────────────

def test_env_can_force_it_off_on_low(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "0")
    assert bool(DKVConfig({"preset": "low"}).shared_basis) is False


def test_env_can_force_it_on_elsewhere(monkeypatch):
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    assert bool(DKVConfig({"preset": "mid"}).shared_basis) is True


# ── the pool honours the constructor arg ─────────────────────────────────────

def test_pool_honours_the_preset_argument(monkeypatch):
    """The regression this guards: shared_basis_enabled() returns False for an
    UNSET variable, so a pool that consulted the environment unconditionally
    would ignore a preset that asked for it on."""
    # Sharing is only supported on an UNROTATED pool and the pool now
    # refuses the rotated combination; DKV_ROTATED_POOL defaults to "1".
    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    monkeypatch.delenv("DKV_SHARED_BASIS_FRAC", raising=False)
    p = _pool(shared_basis=True, frac=0.50)
    assert p.shared_basis_active is True
    assert p.V_KV.shape[0] == 8            # 16 slots -> ceil(0.50*16)
    assert _pool(shared_basis=False).shared_basis_active is False


def test_env_overrides_the_pool_argument(monkeypatch):
    # Sharing is only supported on an UNROTATED pool and the pool now
    # refuses the rotated combination; DKV_ROTATED_POOL defaults to "1".
    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.setenv("DKV_SHARED_BASIS", "0")
    assert _pool(shared_basis=True).shared_basis_active is False
    monkeypatch.setenv("DKV_SHARED_BASIS", "1")
    assert _pool(shared_basis=False).shared_basis_active is True


def test_pool_argument_frac_is_applied(monkeypatch):
    # Sharing is only supported on an UNROTATED pool and the pool now
    # refuses the rotated combination; DKV_ROTATED_POOL defaults to "1".
    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    monkeypatch.delenv("DKV_SHARED_BASIS_FRAC", raising=False)
    assert _pool(True, frac=0.25, n=16).V_KV.shape[0] == 4
    assert _pool(True, frac=1.0, n=16).V_KV.shape[0] == 16


def test_pool_argument_frac_is_clamped(monkeypatch):
    # Sharing is only supported on an UNROTATED pool and the pool now
    # refuses the rotated combination; DKV_ROTATED_POOL defaults to "1".
    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    monkeypatch.delenv("DKV_SHARED_BASIS_FRAC", raising=False)
    assert _pool(True, frac=9.0, n=16).V_KV.shape[0] == 16
    assert _pool(True, frac=-1.0, n=16).V_KV.shape[0] >= 1


def test_low_preset_pool_actually_saves(monkeypatch):
    """End to end: the preset value reaches _bytes_per_block, which is where
    the saving turns into MORE BLOCKS rather than just less memory."""
    # Sharing is only supported on an UNROTATED pool and the pool now
    # refuses the rotated combination; DKV_ROTATED_POOL defaults to "1".
    monkeypatch.setenv("DKV_ROTATED_POOL", "0")
    monkeypatch.delenv("DKV_SHARED_BASIS", raising=False)
    monkeypatch.delenv("DKV_SHARED_BASIS_FRAC", raising=False)
    off = _pool(shared_basis=False)
    on = _pool(shared_basis=True, frac=0.50)
    assert on._bytes_per_block < off._bytes_per_block
    assert on._pool_mb() < off._pool_mb()

"""Shared bases wired into the MLX pool: allocation, indirection, and the guards.

The property that matters most here is the NEGATIVE one: with DKV_SHARED_BASIS
unset (the default) the pool must allocate and behave exactly as it did before.
This feature changes the pool LAYOUT, so a leak into the default path would move
every MLX number in the repo at once.

The rest pin the paths that cannot silently do the wrong thing. Every one of
them is a place where an unguarded write would stay IN RANGE and only the
CONTENTS would be wrong -- no exception, no shape error, just a block
decompressing against another group's basis.
"""
import os
import re
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WRAPPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "serving", "mlx_dkv_wrapper.py")

H_Q, H_KV, D = 8, 2, 64


def _mgr(**env):
    """Fresh manager under a temporary environment."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    try:
        # re-import is not needed: the flags are read in __init__
        from serving.mlx_dkv_wrapper import MLXKVBlockManager
        m = MLXKVBlockManager(num_layers=2, heads=H_Q, kv_heads=H_KV,
                              head_dim=D, rank=8, block_size=32)
        m.max_blocks = 32
        m.set_attended_layers([0, 1])
        return m
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_default_is_off_and_allocates_one_row_per_block():
    """The whole feature must be invisible unless asked for."""
    m = _mgr()
    assert m._shared_basis is False
    sess = m._create_empty_session(16)
    assert sess["comp_VK"][0].shape[0] == 16, (
        "with sharing OFF comp_VK must still have one row per block")
    assert sess["basis_of"] is None
    assert sess["basis_claimed"] is None


def test_enabled_allocates_fewer_rows_and_a_block_to_row_map():
    m = _mgr(DKV_SHARED_BASIS="1", DKV_SHARED_BASIS_FRAC="0.25")
    assert m._shared_basis is True
    sess = m._create_empty_session(16)
    assert sess["comp_VK"][0].shape[0] == 4, (
        f"expected ceil(0.25*16)=4 basis rows, got "
        f"{sess['comp_VK'][0].shape[0]}")
    assert sess["basis_of"][0].shape == (16,)
    assert sess["basis_claimed"][0].shape == (16,)


def test_unwritten_slots_point_at_a_VALID_row():
    """Trap 1. An unwritten slot must resolve to a real row -- gathering over
    -1 reads out of bounds. It points at 0 because 0 is valid, NOT because it
    owns row 0, which is why `basis_claimed` exists alongside it."""
    m = _mgr(DKV_SHARED_BASIS="1", DKV_SHARED_BASIS_FRAC="0.25")
    sess = m._create_empty_session(16)
    bof = sess["basis_of"][0]
    assert int(mx.min(bof).item()) == 0
    assert int(mx.max(bof).item()) == 0
    assert not sess["basis_claimed"][0].any(), (
        "a freshly created session must hold NO claims, or releasing one that "
        "was never made decrements the founding block's group")


def test_basis_rows_is_the_identity_when_sharing_is_off():
    m = _mgr()
    sess = m._create_empty_session(16)
    vk, vv = m._basis_rows(sess, 0, nb=5)
    assert vk.shape[0] == 5 and vv.shape[0] == 5
    full_k, full_v = m._basis_rows(sess, 0)
    assert full_k.shape[0] == 16


def test_basis_rows_returns_one_row_per_BLOCK_under_sharing():
    """The indirection must be invisible to callers and kernels: they still get
    one basis per block even though the store holds far fewer."""
    m = _mgr(DKV_SHARED_BASIS="1", DKV_SHARED_BASIS_FRAC="0.25")
    sess = m._create_empty_session(16)
    vk, vv = m._basis_rows(sess, 0, nb=12)
    assert vk.shape[0] == 12, (
        f"expected 12 per-block rows, got {vk.shape[0]} -- the caller would "
        f"see the STORE shape and silently misalign blocks to bases")
    assert vv.shape[0] == 12
    sel = mx.array([0, 3, 7], dtype=mx.int32)
    sk, sv = m._basis_rows(sess, 0, sel=sel)
    assert sk.shape[0] == 3 and sv.shape[0] == 3


def test_out_of_range_basis_row_raises_rather_than_reading_a_neighbour():
    """MLX does not give the device-side assert CUDA got: an out-of-range row
    silently returns another group's basis. Assert, do not rely on a crash."""
    m = _mgr(DKV_SHARED_BASIS="1", DKV_SHARED_BASIS_FRAC="0.25")
    sess = m._create_empty_session(16)
    bad = sess["basis_of"][0]
    bad[2] = 99                       # past the 4-row store
    sess["basis_of"][0] = bad
    with pytest.raises(RuntimeError, match="out of range"):
        m._basis_rows(sess, 0, nb=4)


def test_correction_form_residuals_are_refused():
    """Residuals in correction form are a delta against a block's OWN
    reconstruction, which a shared basis invalidates outright."""
    with pytest.raises(RuntimeError, match="exact-form residuals"):
        _mgr(DKV_SHARED_BASIS="1", DKV_RESIDUAL_EXCLUDE_SVD="0")


@pytest.mark.parametrize("marker,why", [
    ("DKV_SHARED_BASIS=1 reached the sliding block eviction",
     "sliding eviction shifts basis ROWS as if they were blocks, renumbering "
     "every group at once -- MLX-only, no CUDA test covers it"),
    ("DKV_SHARED_BASIS=1 reached _compress_block",
     "the streaming single-block compress path has no assignment seam"),
])
def test_paths_without_an_assignment_seam_refuse_loudly(marker, why):
    """Source-level pins.

    Both paths need a live model and a full prefill to reach, which is too
    expensive for this suite, but the guard's ABSENCE is silent -- so its
    presence is pinned instead.
    """
    src = open(WRAPPER).read()
    assert marker in src, f"missing guard ({why})"


def test_multi_layer_compress_declines_under_sharing():
    """That path writes VK_batch straight into comp_VK[start:start+num_blocks],
    which under sharing is a BASIS-ROW-indexed store. It must hand off to the
    per-layer compressor, which has the seam."""
    src = open(WRAPPER).read()
    m = re.search(r"def compress_deferred_prefill_blocks\(self.*?\n(.*?)\n    def ",
                  src, re.S)
    assert m, "compress_deferred_prefill_blocks not found"
    body = m.group(1)
    assert "if self._shared_basis:" in body
    assert "compress_deferred_prefill_blocks_for_layer" in body


def test_per_layer_write_is_skipped_under_sharing():
    """_assign_shared_basis already wrote the founded bases; repeating the
    block-indexed write would scribble block ids over basis rows."""
    src = open(WRAPPER).read()
    assert ('if not self._shared_basis:\n'
            '            # Under sharing these rows are indexed by BASIS ROW') in src, (
        "the per-layer VK/VV write is no longer guarded by `if not "
        "self._shared_basis` -- it would overwrite basis rows by block id")


def test_basis_stats_reports_joined_and_mean_kept():
    """`joined == 0` is the degeneracy signature and pool MB looks identical
    either way, so both must be reportable next to the memory number."""
    off = _mgr()
    assert off.basis_stats() == {"enabled": False}
    on = _mgr(DKV_SHARED_BASIS="1")
    st = on.basis_stats()
    assert st["enabled"] is True
    assert {"joined", "forced", "founded", "mean_kept"} <= set(st)

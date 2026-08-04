"""Exact residual rows belong in the DENSE half, as MLX puts them.

MLX carries a block's exact residual rows in the DENSE partition
(`mlx_dkv_wrapper.py:1031  dense_k_for_attn = concat([res_k_all, dense_k])`) and
masks their lossy low-rank twins out of the SPARSE half
(`:771  delta_s = where(res_mask, -inf, delta_s)`). CUDA keeps them in the SPARSE
half, which is what puts `DKV_SPARSE_BIAS=auto` on the wrong side of
`(lse_dense - lse_sparse)` -- see `residuals_in_dense.__doc__`.

The invariant these tests pin is the one that makes the change safe to land:
a residual token must appear EXACTLY ONCE. Score it in the sparse half AND emit
it as a dense row and it is double-counted (the F1 bug); mask it in sparse and
fail to emit it and the token vanishes from attention entirely. So the kernel
flag and the row builder are two halves of one change and must move together.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_core.sparse_decode.triton_fused_decode import (      # noqa: E402
    build_dense_residual_rows, residuals_in_dense,
)

N, MAX_RES, H, D = 3, 4, 2, 8


def _g(k_pos, v_pos=None):
    """A gather dict shaped like `_gather_routed_blocks_for_kernel`'s output."""
    gen = torch.Generator().manual_seed(11)
    kp = torch.tensor(k_pos, dtype=torch.int16)
    vp = torch.tensor(v_pos if v_pos is not None else k_pos, dtype=torch.int16)
    return {
        "has_res": True,
        "res_pos": kp, "res_pos_v": vp,
        "res_k": torch.randn(N, MAX_RES, H, D, generator=gen),
        "res_v": torch.randn(N, MAX_RES, H, D, generator=gen),
        "anchors_K": torch.randn(N, H, D, generator=gen),
        "anchors_V": torch.randn(N, H, D, generator=gen),
    }


def test_default_is_off():
    """Half-landed, this change drops tokens. It must not be on by accident."""
    os.environ.pop("DKV_RESIDUALS_IN_DENSE", None)
    assert residuals_in_dense() is False


def test_row_count_matches_valid_slots_only():
    """-1 is padding; padded slots must not become attended rows."""
    g = _g([[0, 5, -1, -1], [3, -1, -1, -1], [-1, -1, -1, -1]])
    K, V, n = build_dense_residual_rows(g)
    assert n == 3, f"expected 3 valid slots, got {n}"
    assert K.shape == (3, H, D) and V.shape == (3, H, D)


def test_row_is_anchor_plus_residual():
    """CUDA stores residuals ANCHOR-RELATIVE, so the exact key is anchor + res --
    the reconstruction probe_residual_values verified at cos 1.0000."""
    g = _g([[7, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    K, V, n = build_dense_residual_rows(g)
    assert n == 1
    assert torch.allclose(K[0], g["anchors_K"][0] + g["res_k"][0, 0], atol=1e-5)
    assert torch.allclose(V[0], g["anchors_V"][0] + g["res_v"][0, 0], atol=1e-5)


def test_no_residuals_returns_nothing():
    g = _g([[-1] * MAX_RES] * N)
    K, V, n = build_dense_residual_rows(g)
    assert (K, V, n) == (None, None, 0)
    assert build_dense_residual_rows({"has_res": False}) == (None, None, 0)


def test_k_only_slot_falls_back_to_anchor_for_v():
    """K and V positions are chosen independently. A slot exact in K but not V must
    still emit a row -- with its TRUE K and the anchor's V, which is what the
    sparse-half substitution did. Dropping it would lose an exact key."""
    g = _g(k_pos=[[2, -1, -1, -1], [-1] * 4, [-1] * 4],
           v_pos=[[-1, -1, -1, -1], [-1] * 4, [-1] * 4])
    K, V, n = build_dense_residual_rows(g)
    assert n == 1, "a K-only slot must still produce a row"
    assert torch.allclose(K[0], g["anchors_K"][0] + g["res_k"][0, 0], atol=1e-5)
    assert torch.allclose(V[0], g["anchors_V"][0], atol=1e-5), \
        "V should fall back to the anchor when the slot has no V residual"


def test_union_of_k_and_v_positions():
    """Rows are emitted on the UNION of the two position sets, so a token exact in
    only one of K/V is never silently dropped."""
    g = _g(k_pos=[[1, -1, -1, -1], [-1] * 4, [-1] * 4],
           v_pos=[[-1, 2, -1, -1], [-1] * 4, [-1] * 4])
    _, _, n = build_dense_residual_rows(g)
    assert n == 2, f"union of one K slot and one V slot should be 2 rows, got {n}"


def test_rows_are_finite_and_dtype_honoured():
    g = _g([[0, 1, -1, -1], [2, -1, -1, -1], [-1] * 4])
    K, V, _ = build_dense_residual_rows(g, dtype=torch.float16)
    assert K.dtype == torch.float16 and V.dtype == torch.float16
    assert torch.isfinite(K).all() and torch.isfinite(V).all()

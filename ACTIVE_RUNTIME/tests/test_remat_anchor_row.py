"""A materialised block must include its ANCHOR as a row.

The anchor is a real token, not a reference point: a block stores
`anchor = k[..., 0]` and `active = k[..., 1:]` (streaming_sparse_ingest.py:1206).
Both references attend it as its own row --

    MLX     full_k = concatenate([ak_e, ak_e + delta_k], axis=2)
                                                (mlx_dkv_wrapper.py:4053)
    Triton  p_anchor = exp(s_anchor - m_new); l_i += p_anchor + p_delta_sum
                                                (triton_fused_decode.py:751)

-- so a block contributes 1 + seq_len rows. reconstruct_blocks returned only the
S delta rows, folding the anchor into each and dropping the anchor token itself:
16 real tokens at K=16, 122 when attending all blocks. That made remat quietly
NOT MLX's decode form, so using it to test "materialise vs project-then-attend"
compared two differences at once.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_core.sparse_decode.remat_cache import (      # noqa: E402
    reconstruct_blocks, attend_with_remat,
)

N, S, R, H, D = 3, 5, 4, 2, 8


def _inputs(zero_lowrank=False):
    g = torch.Generator().manual_seed(7)
    U = torch.zeros(N, S, R) if zero_lowrank else torch.randn(N, S, R, generator=g)
    return dict(
        U=U,
        V_K=torch.randn(N, R, H, D, generator=g),
        V_V=torch.randn(N, R, H, D, generator=g),
        anchors_K=torch.randn(N, H, D, generator=g),
        anchors_V=torch.randn(N, H, D, generator=g),
        scales=torch.ones(N),
        U_scale=torch.ones(N),
        rank=R,
    )


def test_materialised_block_has_one_extra_row_for_the_anchor():
    kw = _inputs()
    K, V = reconstruct_blocks(**kw)
    assert K.shape == (N, S + 1, H, D), (
        f"expected 1 + {S} rows per block (anchor + active), got {K.shape} -- "
        "the anchor token is being dropped from attention")
    assert V.shape == (N, S + 1, H, D), V.shape


def test_row_zero_is_exactly_the_anchor():
    """Not anchor+delta: the anchor row is the anchor key itself."""
    kw = _inputs()
    K, V = reconstruct_blocks(**kw)
    assert torch.allclose(K[:, 0], kw["anchors_K"], atol=1e-5), \
        "row 0 is not the bare anchor key"
    assert torch.allclose(V[:, 0], kw["anchors_V"], atol=1e-5), \
        "row 0 is not the bare anchor value"


def test_active_rows_still_carry_the_low_rank_delta():
    """With U=0 every active row collapses to the anchor; with U!=0 it must not.

    Guards against 'fixing' the shape by padding with a duplicate anchor.
    """
    K0, _ = reconstruct_blocks(**_inputs(zero_lowrank=True))
    for s in range(1, S + 1):
        assert torch.allclose(K0[:, s], K0[:, 0], atol=1e-5), \
            "with U=0 an active row should equal the anchor"
    K1, _ = reconstruct_blocks(**_inputs())
    assert not torch.allclose(K1[:, 1], K1[:, 0], atol=1e-4), \
        "active rows lost their low-rank delta"


def test_attend_masks_anchor_plus_seq_len_rows():
    """seq_lens counts ACTIVE tokens, so 1 + seq_lens rows are live.

    Bounding by seq_lens instead would drop each block's last real token.
    """
    kw = _inputs()
    K, V = reconstruct_blocks(**kw)
    q = torch.randn(1, H, 1, D)
    seq = torch.tensor([S, 1, 0])          # full, one active, anchor-only
    out = attend_with_remat(q, K, V, seq, None, None, 0, 1)
    assert out.shape == (1, H, 1, D), out.shape
    assert torch.isfinite(out).all(), "non-finite output"
    # A block with seq_len=0 still has its anchor live, so nothing is fully
    # masked and the softmax denominator can never be zero.
    out2 = attend_with_remat(q, K, V, torch.zeros(N, dtype=torch.long),
                             None, None, 0, 1)
    assert torch.isfinite(out2).all(), \
        "all-zero seq_lens produced non-finite output -- the anchor row should " \
        "still be attended"

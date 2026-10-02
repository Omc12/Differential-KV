"""DKV_PREFILL_LOWMEM must not change the arithmetic of prefill history attention.

The opt-in path (a) computes the V residual correction as a contraction rather
than broadcast-then-sum, and (b) slices the chunk's queries. Both are exact
rewrites; this pins them against the default path on random inputs with
residuals present, in both residual semantics.
"""
import pytest
import torch

from native_core.sparse_decode.triton_fused_decode import (
    _prefill_fused_history_attend_compiled as attend,
)


def _inputs(seed=0, N=5, S=15, R=4, H=3, Q=37, D=8, P=6, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    pos = torch.randint(0, S, (N, P), generator=g)
    pos[0, -1] = -1                                   # an invalid slot, masked
    return dict(
        U=r(N, S, R), V_K=r(N, R, H, D), V_V=r(N, R, H, D),
        anchors_K=r(N, H, D), anchors_V=r(N, H, D),
        scales=r(N).abs() + 0.5,
        cos_sliced=r(N, 1 + S, 1, D), sin_sliced=r(N, 1 + S, 1, D),
        q=r(1, H, Q, D),
        seq_lens=torch.randint(1, S + 1, (N,), generator=g),
        inv_scale=D ** -0.5,
        residual_K_positions=pos, residual_K_values=r(N, P, H, D),
        residual_V_positions=pos.clone(), residual_V_values=r(N, P, H, D),
    )


@pytest.mark.parametrize("exact", [False, True])
def test_contraction_matches_broadcast(exact):
    kw = _inputs()
    ref = attend(exact_residual=exact, **kw)
    got = attend(exact_residual=exact, lowmem=True, **kw)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("qs", [1, 8, 16, 37])
def test_query_slicing_matches_whole(exact, qs):
    kw = _inputs(seed=1)
    q = kw.pop("q")
    ref = attend(q=q, exact_residual=exact, **kw)
    got = torch.cat([attend(q=q[:, :, s:s + qs], exact_residual=exact,
                            lowmem=True, **kw)
                     for s in range(0, q.shape[2], qs)], dim=3)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("tile", [1, 2, 3, 5, 0])
def test_block_tiling_matches_whole(exact, tile):
    """Attending the compressed blocks a tile at a time and merging by
    log-sum-exp must equal attending them all at once."""
    from native_core.sparse_decode.triton_fused_decode import (
        history_attend_block_tiled)
    kw = _inputs(seed=2, N=7)
    q = kw.pop("q")
    ref = attend(q=q, exact_residual=exact, lowmem=True, **kw)
    got = history_attend_block_tiled(
        lambda k: attend(q=q, exact_residual=exact, lowmem=True, **k),
        kw, tile, torch.float32)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

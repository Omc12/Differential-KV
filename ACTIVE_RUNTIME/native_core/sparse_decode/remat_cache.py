"""Re-materialisation cache for CUDA decode — the MLX `DKV_DECODE_CACHE` port.

THE PROBLEM THIS SOLVES
-----------------------
CUDA rebuilds `anchor + (U @ V) * scale` for every routed block on EVERY decoded
token, in EVERY layer. At K=16 routed blocks of 257 tokens over 28 layers that is
115k token-reconstructions per generated token. Measured: ~64 ms/token of decode
GPU time, against dense's 9.3 ms/token TOTAL for more attended tokens.

MLX does not do this. `mlx_dkv_wrapper`'s DKV_DECODE_CACHE re-routes and
re-materialises the selected blocks once every DKV_DECODE_CACHE_INTERVAL tokens
(default 16) and reuses the result in between, trading a bounded staleness for
~16x less reconstruction work. MLX lands at ~75% of dense decode speed; CUDA at
~9%. That gap is this cache.

NOT to be confused with `DKV_DECODE_CACHE_CUDA` in dkv_attention.py, which caches
only the gather/index-select for an unchanged block set. That is the cheap part —
an experiment removing gather allocations entirely (DKV_STATIC_GATHER) measured
0% — while re-materialisation is the expensive part and had no cache at all.

STALENESS CONTRACT
------------------
Between refreshes the ROUTED CONTENT is frozen: blocks selected at refresh time
keep being attended with the K/V reconstructed then. New tokens still enter
through the DENSE window, which is never cached, so recent context is always
exact. This is the same trade MLX makes.

Consequences that MUST be gated by the needle DEPTH sweep
(colab/validate_cuda_dkv.py), because a needle whose block is routed only AFTER
a refresh boundary is invisible until the next one:
  * refresh on interval boundaries AND on any routing-version change
  * refresh on pool writes (a flushed block changes what reconstruction means)

STATUS: opt-in (DKV_REMAT_CACHE=1), default OFF, NOT validated on GPU.
The math below is verified on CPU against a direct reference — run this file
directly to check. Correctness of the *caching policy* is what needs hardware.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch


def remat_enabled() -> bool:
    """DKV_REMAT_CACHE — deliberately a NEW name.

    Not DKV_DECODE_CACHE (MLX's, different runtime) and not
    DKV_DECODE_CACHE_CUDA (the existing narrow gather cache). Four separate
    knob-name collisions have already been found in this project where a
    same-sounding CUDA flag was a different, narrower feature with a different
    default; this does not add a fifth.
    """
    return os.environ.get("DKV_REMAT_CACHE", "0") == "1"


def remat_interval() -> int:
    """Tokens between refreshes. MLX's DKV_DECODE_CACHE_INTERVAL default is 16."""
    try:
        return max(1, int(os.environ.get("DKV_REMAT_INTERVAL", "16")))
    except ValueError:
        return 16


def _scatter_residuals(X: torch.Tensor, res_val: torch.Tensor,
                       res_pos: torch.Tensor) -> torch.Tensor:
    """Add the sparse exact-value corrections into a materialised [N, S, H, D].

    `residual_*_values` hold `exact - lowrank_recon` at the worst-reconstructed
    positions (lowrank.py:659), so they are ADDED to the low-rank twin and never
    replace it — the CORRECTION form that `validate_cuda_dkv.py` test 2 gates on.
    Adding is also what the two reference decoders do: the Triton kernel's C1
    block (`s = tl.where(offs_s == r_pos_k, s + r_corr, s)`) and
    `_pytorch_vectorized_sparse_attn_decode`'s `scatter_add_`.

    Positions are within-block offsets, `-1` padded. Out-of-range entries are
    DROPPED rather than clamped: clamping would fold a stray offset into the last
    live row and silently corrupt it, whereas the Triton kernel's `offs_s ==
    r_pos_k` comparison simply never fires. Dropping matches that.

    No `+1` on the index. `_pytorch_vectorized_sparse_attn_decode` offsets by one
    because it prepends a bare-anchor row at index 0; this layout (like Triton's)
    folds the anchor into every row instead, so offset `p` is row `p`.
    """
    if res_val is None or res_pos is None or res_val.numel() == 0:
        return X
    N, S, H, D = X.shape
    pos = res_pos.long()
    keep = (pos >= 0) & (pos < S)
    # No `if not keep.any()` early-out: that is bool() on a device tensor, i.e. a
    # sync, and it ran twice per reconstruct (K and V). The recorder put it at
    # 10,752 hits -- the single largest site in the whole run, and self-inflicted.
    # Masked-out rows contribute an exact zero to scatter_add_ at a clamped-but-
    # valid index, so skipping the work saves nothing and costs a pipeline drain.
    index = pos.clamp(0, S - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
    src = res_val.to(X.dtype) * keep.unsqueeze(-1).unsqueeze(-1).to(X.dtype)
    return X.scatter_add_(dim=1, index=index, src=src)


def reconstruct_blocks(
    U: torch.Tensor,            # [N, S, R]      int8 or float
    V_K: torch.Tensor,          # [N, R, H, D]
    V_V: torch.Tensor,          # [N, R, H, D]
    anchors_K: torch.Tensor,    # [N, H, D]
    anchors_V: torch.Tensor,    # [N, H, D]
    scales: torch.Tensor,       # [N]
    U_scale: torch.Tensor,      # [N]
    rank: int,
    res_k: Optional[torch.Tensor] = None,       # [N, MAX_RES, H, D] pre-rotated
    res_pos: Optional[torch.Tensor] = None,     # [N, MAX_RES]  -1 padded
    res_v: Optional[torch.Tensor] = None,       # [N, MAX_RES, H, D]
    res_pos_v: Optional[torch.Tensor] = None,   # [N, MAX_RES]  -1 padded
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise routed blocks to dense K/V.

        K[n, s] = anchors_K[n] + (U[n, s, :rank] @ V_K[n, :rank]) * scales[n]
                  + residual_K[n, s]        (sparse, where present)

    Returns K, V each [N, S, H, D].

    `rank` is the LAYER's active rank, which is <= V_K.shape[1] (the pool's
    allocation width). Slicing to it matters: the pool's columns beyond the
    layer's rank were never written by this layer, and including them adds
    another block's basis — the same class of bug as the Triton rank-mask fix.

    THE RESIDUALS ARE NOT OPTIONAL FOR CORRECTNESS. Without them a routed block
    is attended at pure low-rank fidelity, which has a documented recall floor —
    the residuals are precisely the exact-value correction for the tokens the SVD
    reconstructs worst, which is what a random alphanumeric code is. Omitting
    them costs exact recall while still producing fluent, plausible output: the
    8k needle came back as `ZEBRA-1234` instead of `ZEBRA-4471-QUARTZ`, prefix
    right and exact suffix invented. Pass them whenever the gather reports
    `has_res`.

    K and V take SEPARATE position arrays. `residual_V_positions` are chosen
    independently of `residual_K_positions` (the worst-reconstructed rows differ
    between the two), so using one for both silently corrects the wrong tokens.
    """
    N, S, _ = U.shape
    H, D = V_K.shape[2], V_K.shape[3]
    r = min(int(rank), V_K.shape[1], U.shape[2])

    Uf = U[:, :, :r].float()
    if U_scale is not None:
        Uf = Uf * U_scale.view(N, 1, 1).float()

    # [N, S, r] @ [N, r, H*D] -> [N, S, H*D]
    K = torch.bmm(Uf, V_K[:, :r].reshape(N, r, H * D).float())
    V = torch.bmm(Uf, V_V[:, :r].reshape(N, r, H * D).float())
    K = K.reshape(N, S, H, D) * scales.view(N, 1, 1, 1).float() + anchors_K.unsqueeze(1).float()
    V = V.reshape(N, S, H, D) * scales.view(N, 1, 1, 1).float() + anchors_V.unsqueeze(1).float()

    # Applied in fp32, before the cast back: the corrections are small deltas on
    # top of a much larger anchor, which is exactly where fp16 rounding eats them.
    K = _scatter_residuals(K, res_k, res_pos)
    V = _scatter_residuals(V, res_v, res_pos_v)
    return K.to(V_K.dtype), V.to(V_V.dtype)


class RematCache:
    """Per-(session, layer) materialised K/V with an explicit refresh policy.

    Keyed on (layer, routing_version, pool_generation, step // interval) so it
    refreshes when ANY of: the interval elapses, routing changes, or the pool is
    written. The last two are correctness requirements, not optimisations — a
    stale entry after either would attend to blocks that no longer mean what they
    meant when materialised.
    """

    def __init__(self) -> None:
        self._store: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(layer_idx: int, routing_version: int, pool_generation: int,
                 step: int, interval: Optional[int] = None) -> tuple:
        iv = interval if interval is not None else remat_interval()
        return (layer_idx, routing_version, pool_generation, step // max(1, iv))

    def get(self, key: tuple):
        v = self._store.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key: tuple, K: torch.Tensor, V: torch.Tensor) -> None:
        # Only ever hold one entry per layer: an older interval/version can never
        # be reused, and keeping it would pin N*S*H*D of VRAM per stale entry.
        layer = key[0]
        for k in [k for k in self._store if k[0] == layer and k != key]:
            del self._store[k]
        self._store[key] = (K, V)

    def invalidate_layer(self, layer_idx: int) -> None:
        for k in [k for k in self._store if k[0] == layer_idx]:
            del self._store[k]

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> str:
        tot = self.hits + self.misses
        return (f"remat cache: {self.hits} hits / {tot} "
                f"({100.0 * self.hits / tot:.0f}% reuse)" if tot else "remat cache: unused")


def attend_with_remat(
    q: torch.Tensor,            # [1, H_q, 1, D]
    K_mat: torch.Tensor,        # [N, S, H_kv, D]  materialised routed blocks
    V_mat: torch.Tensor,        # [N, S, H_kv, D]
    seq_lens: torch.Tensor,     # [N]  live tokens per block (rest is padding)
    dense_k: Optional[torch.Tensor],   # [1, H_kv, L, D] pre-rotated dense window
    dense_v: Optional[torch.Tensor],
    dense_len: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """One plain attention over [materialised routed blocks | dense window].

    This is what makes the cache pay: with K/V already materialised there is no
    reconstruction left in the inner loop, so the step is an ordinary SDPA —
    exactly the operation dense does at 9.3 ms/token.

    Padding MUST be masked. Blocks are S_max wide but only seq_lens[n] tokens are
    live; the tail is whatever the pool last held there. Attending it is the same
    class of bug as the dense-window trimming found earlier, and it is silent.
    """
    N, S, H_kv, D = K_mat.shape
    H_q = q.shape[1]
    dev, dt = q.device, q.dtype

    valid = (torch.arange(S, device=dev).view(1, S) <
             seq_lens.to(dev).view(N, 1))                       # [N, S]
    k_parts = [K_mat.reshape(N * S, H_kv, D)]
    v_parts = [V_mat.reshape(N * S, H_kv, D)]
    mask_parts = [valid.reshape(N * S)]

    if dense_k is not None and dense_len > 0:
        k_parts.append(dense_k[0, :, :dense_len].permute(1, 0, 2))
        v_parts.append(dense_v[0, :, :dense_len].permute(1, 0, 2))
        mask_parts.append(torch.ones(dense_len, dtype=torch.bool, device=dev))

    K_all = torch.cat(k_parts, dim=0).to(dt)                     # [T, H_kv, D]
    V_all = torch.cat(v_parts, dim=0).to(dt)
    keep = torch.cat(mask_parts, dim=0)                          # [T]

    K_all = K_all.permute(1, 0, 2).unsqueeze(0)                  # [1, H_kv, T, D]
    V_all = V_all.permute(1, 0, 2).unsqueeze(0)
    if num_key_value_groups > 1:
        K_all = K_all.repeat_interleave(num_key_value_groups, dim=1)
        V_all = V_all.repeat_interleave(num_key_value_groups, dim=1)

    attn_mask = torch.zeros(1, 1, 1, keep.shape[0], device=dev, dtype=dt)
    attn_mask.masked_fill_(~keep.view(1, 1, 1, -1), float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(
        q, K_all, V_all, attn_mask=attn_mask)                    # [1, H_q, 1, D]


if __name__ == "__main__":
    # CPU self-test: the reconstruction must match a direct per-block reference,
    # and the cache must refresh on interval, routing change and pool write.
    torch.manual_seed(0)
    N, S, R_pool, H, D, rank = 4, 16, 48, 2, 64, 24
    U = torch.randn(N, S, R_pool)
    V_K = torch.randn(N, R_pool, H, D)
    V_V = torch.randn(N, R_pool, H, D)
    aK = torch.randn(N, H, D)
    aV = torch.randn(N, H, D)
    sc = torch.rand(N) + 0.5
    us = torch.rand(N) + 0.5

    K, V = reconstruct_blocks(U, V_K, V_V, aK, aV, sc, us, rank)

    ok = True
    for n in range(N):
        for s in range(S):
            ref = (U[n, s, :rank] * us[n]) @ V_K[n, :rank].reshape(rank, H * D)
            ref = ref.reshape(H, D) * sc[n] + aK[n]
            if not torch.allclose(ref, K[n, s], atol=1e-3):
                ok = False
                break
    print(f"reconstruction matches per-block reference : {ok}")
    print(f"K shape {tuple(K.shape)}  V shape {tuple(V.shape)}  (expect ({N}, {S}, {H}, {D}))")

    # ── Residual corrections ────────────────────────────────────────────────
    # The regression this guards: dropping these attends every routed block at
    # pure low-rank fidelity, which reads fluently and fails exact recall.
    MAX_RES = 3
    res_pos = torch.full((N, MAX_RES), -1, dtype=torch.int16)
    res_pos[:, 0] = 2                      # a live correction in every block
    res_pos[0, 1] = 7                      # a second one in block 0
    res_pos[1, 1] = S + 5                  # out of range -> must be DROPPED
    res_pos_v = torch.full((N, MAX_RES), -1, dtype=torch.int16)
    res_pos_v[:, 0] = 5                    # V positions differ from K's, by design
    res_k = torch.randn(N, MAX_RES, H, D)
    res_v = torch.randn(N, MAX_RES, H, D)

    Kr, Vr = reconstruct_blocks(U, V_K, V_V, aK, aV, sc, us, rank,
                                res_k=res_k, res_pos=res_pos,
                                res_v=res_v, res_pos_v=res_pos_v)

    exp_K, exp_V = K.clone(), V.clone()
    exp_K[:, 2] += res_k[:, 0]
    exp_K[0, 7] += res_k[0, 1]
    exp_V[:, 5] += res_v[:, 0]
    print(f"K residuals land at the right rows        : "
          f"{torch.allclose(Kr, exp_K, atol=1e-3)}   (want True)")
    print(f"V uses its OWN positions, not K's         : "
          f"{torch.allclose(Vr, exp_V, atol=1e-3)}   (want True)")
    print(f"out-of-range residual dropped, not folded : "
          f"{torch.allclose(Kr[1, S - 1], K[1, S - 1], atol=1e-3)}   (want True)")
    print(f"-1 padding contributes nothing            : "
          f"{torch.allclose(Kr[3, 9], K[3, 9], atol=1e-3)}   (want True)")
    print(f"residuals actually change the result      : "
          f"{not torch.allclose(Kr, K, atol=1e-3)}   (want True)")
    K2, _ = reconstruct_blocks(U, V_K, V_V, aK, aV, sc, us, rank)
    print(f"no-residual call unchanged (default path) : "
          f"{torch.allclose(K2, K, atol=1e-6)}   (want True)")

    # ── attend_with_remat ───────────────────────────────────────────────────
    # Padding past seq_lens must not be attended: the tail of a partially filled
    # block is whatever the pool last held there.
    H_q, groups = H * 2, 2
    q = torch.randn(1, H_q, 1, D)
    seq_lens = torch.tensor([S, S // 2, S, 1])
    dense_k = torch.randn(1, H, 4, D)
    dense_v = torch.randn(1, H, 4, D)
    o1 = attend_with_remat(q, Kr, Vr, seq_lens, dense_k, dense_v, 4, groups)
    Kg, Vg = Kr.clone(), Vr.clone()
    Kg[1, S // 2:] = 1e4                   # garbage in the padded tail
    Vg[1, S // 2:] = 1e4
    Kg[3, 1:] = 1e4
    Vg[3, 1:] = 1e4
    o2 = attend_with_remat(q, Kg, Vg, seq_lens, dense_k, dense_v, 4, groups)
    print(f"padding masked (garbage tail ignored)     : "
          f"{torch.allclose(o1, o2, atol=1e-4)}   (want True)")
    print(f"attend output shape {tuple(o1.shape)}  (expect (1, {H_q}, 1, {D}))")

    c = RematCache()
    k0 = RematCache.make_key(3, routing_version=1, pool_generation=7, step=0, interval=16)
    k1 = RematCache.make_key(3, routing_version=1, pool_generation=7, step=15, interval=16)
    k2 = RematCache.make_key(3, routing_version=1, pool_generation=7, step=16, interval=16)
    k3 = RematCache.make_key(3, routing_version=2, pool_generation=7, step=0, interval=16)
    k4 = RematCache.make_key(3, routing_version=1, pool_generation=8, step=0, interval=16)
    print(f"same interval reuses (step 0 vs 15)   : {k0 == k1}   (want True)")
    print(f"interval boundary refreshes (0 vs 16) : {k0 != k2}   (want True)")
    print(f"routing change refreshes              : {k0 != k3}   (want True)")
    print(f"pool write refreshes                  : {k0 != k4}   (want True)")

    c.put(k0, K, V)
    print(f"hit on same key                       : {c.get(k0) is not None}   (want True)")
    c.put(k2, K, V)
    print(f"stale entry evicted, not accumulated  : {c.get(k0) is None}   (want True)")

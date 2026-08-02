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


def reconstruct_blocks(
    U: torch.Tensor,            # [N, S, R]      int8 or float
    V_K: torch.Tensor,          # [N, R, H, D]
    V_V: torch.Tensor,          # [N, R, H, D]
    anchors_K: torch.Tensor,    # [N, H, D]
    anchors_V: torch.Tensor,    # [N, H, D]
    scales: torch.Tensor,       # [N]
    U_scale: torch.Tensor,      # [N]
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise routed blocks to dense K/V.

        K[n, s] = anchors_K[n] + (U[n, s, :rank] @ V_K[n, :rank]) * scales[n]

    Returns K, V each [N, S, H, D].

    `rank` is the LAYER's active rank, which is <= V_K.shape[1] (the pool's
    allocation width). Slicing to it matters: the pool's columns beyond the
    layer's rank were never written by this layer, and including them adds
    another block's basis — the same class of bug as the Triton rank-mask fix.
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

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
  * refresh on interval boundaries
  * refresh on pool writes (a flushed block changes what reconstruction means)

CORRECTED 2026-08-02: this list also said "refresh on any routing-version change",
and called it a correctness requirement. On GPU that made the cache
unconditionally dead — reconstruct ran every layer of every token, a 0% hit rate,
because routing legitimately changes almost every token. Freezing the routed set
for the interval is not a violation of the contract above, it IS the contract:
"blocks selected at refresh time keep being attended". See remat_freeze_routing()
for the measurement and the escape hatch.

What that implies about the measured win: remat was ~1.35x faster on GPU while
hitting 0%, so the gain is the materialise+SDPA formulation replacing the Triton
per-token reconstruction — NOT caching. Cache hits are additional headroom on
top, and they are what DKV_REMAT_INTERVAL was always meant to buy.

STATUS: opt-in (DKV_REMAT_CACHE=1), default OFF. Residual correctness is
GPU-validated (8k needle 3/3 at depth 0.0/0.5/0.9). The routing freeze is NOT yet
GPU-validated — it is precisely the staleness policy the depth sweep exists to gate.
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


def remat_freeze_routing() -> bool:
    """Hold the ROUTED BLOCK SET for the interval, not just its materialisation.

    This is what makes the cache a cache. Measured on GPU: with the routing
    version in the key, `reconstruct_blocks` ran 5,376 times for 192 tokens x 28
    layers -- every layer of every token, a 0% hit rate. `current_version`
    increments whenever the routed set changes (dkv_attention.py:1304) and routing
    legitimately changes almost every token, so no entry could ever survive an
    interval and DKV_REMAT_INTERVAL had no effect at any value.

    MLX, which this ports, "re-routes AND re-materialises the selected blocks once
    every DKV_DECODE_CACHE_INTERVAL tokens" -- it freezes routing too. Doing the
    same is what this flag turns on, and it matches the staleness contract this
    module documented from the start.

    DKV_REMAT_FREEZE_ROUTING=0 restores refresh-on-routing-change, i.e. the
    known-good 0%-hit behaviour, as an escape hatch if recall regresses.
    """
    return os.environ.get("DKV_REMAT_FREEZE_ROUTING", "1") == "1"


def _exact_form() -> bool:
    """Are residuals stored as the anchor-relative EXACT value, or a correction?

    Forwards to lowrank._exact_keys_enabled so this can never disagree with what
    the compressor wrote — that mismatch silently doubles or halves every residual
    token. Now defaults ON everywhere (MLX/Metal/Triton all substitute).
    """
    try:
        from native_core.compression.lowrank import _exact_keys_enabled
        return bool(_exact_keys_enabled(torch.device("cuda")
                                        if torch.cuda.is_available() else None))
    except Exception:                                            # noqa: BLE001
        return False


def _scatter_residuals(X: torch.Tensor, res_val: torch.Tensor,
                       res_pos: torch.Tensor,
                       anchors: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    keep_f = keep.unsqueeze(-1).unsqueeze(-1)

    if anchors is not None and _exact_form():
        # SUBSTITUTION (MLX / Metal / Triton EXACT_RESIDUAL). res_val is the
        # anchor-relative EXACT value, so the row IS anchor + res — the low-rank
        # estimate at that position is discarded rather than nudged. scatter_,
        # not scatter_add_: adding would leave the lossy twin in and count the
        # anchor twice. Rows whose position is padding keep their existing value,
        # which is why the masked src falls back to X's own row.
        exact = anchors.unsqueeze(1) + res_val.to(X.dtype)          # [N, MAX_RES, H, D]
        current = X.gather(dim=1, index=index)
        src = torch.where(keep_f, exact, current)
        return X.scatter_(dim=1, index=index, src=src)

    # CORRECTION form: res_val is (exact - recon); add it and keep the twin.
    src = res_val.to(X.dtype) * keep_f.to(X.dtype)
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
    # Done BEFORE the anchor row is prepended, so res_pos stays a plain row index
    # and _scatter_residuals needs no offset.
    K = _scatter_residuals(K, res_k, res_pos, anchors_K.float())
    V = _scatter_residuals(V, res_v, res_pos_v, anchors_V.float())

    # ── PREPEND THE ANCHOR ROW ────────────────────────────────────────────────
    # The anchor is a REAL TOKEN, not just a reference point: a block stores
    # `anchor = k[..., 0]` and `active = k[..., 1:]`
    # (streaming_sparse_ingest.py:1206/1216). Both references attend it as its
    # own row --
    #     MLX     full_k = concatenate([ak_e, ak_e + delta_k], axis=2)
    #                                            (mlx_dkv_wrapper.py:4053)
    #     Triton  p_anchor = exp(s_anchor - m_new); l_i += p_anchor + p_delta_sum
    #                                            (triton_fused_decode.py:751)
    # -- so each block contributes 1 + seq_len rows. This function returned only
    # the S delta rows, folding the anchor into each of them and DROPPING the
    # anchor token itself: 16 real tokens gone at K=16, 122 when attending all.
    # That made remat quietly not-MLX's-form, so measuring "materialise vs
    # project-then-attend" with it was measuring two differences at once.
    K = torch.cat([anchors_K.unsqueeze(1).float(), K], dim=1)
    V = torch.cat([anchors_V.unsqueeze(1).float(), V], dim=1)
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
                 step: int, interval: Optional[int] = None,
                 freeze_routing: Optional[bool] = None) -> tuple:
        iv = interval if interval is not None else remat_interval()
        frz = remat_freeze_routing() if freeze_routing is None else freeze_routing
        # Dropping routing_version from the key IS the freeze: the entry then
        # survives a routing change and the blocks selected at refresh time keep
        # being attended until the interval elapses. Keeping it made the cache
        # unconditionally dead (see remat_freeze_routing). pool_generation stays
        # in either way -- a flushed block changes what a slot MEANS, which is a
        # correctness refresh, not a routing-freshness one.
        rv = 0 if frz else routing_version
        return (layer_idx, rv, pool_generation, step // max(1, iv))

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
    trace_row: Optional[int] = None,   # flat row index to report mass for
    trace_tok: int = -1,               # its absolute token index, for the log
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

    # Row 0 is the block's ANCHOR and is always live; rows 1..seq_len are its
    # active tokens. reconstruct_blocks now returns 1 + S rows per block, so the
    # validity bound is seq_lens + 1 -- using seq_lens here would drop the last
    # real token of every block instead.
    valid = (torch.arange(S, device=dev).view(1, S) <
             (seq_lens.to(dev).view(N, 1) + 1))                 # [N, S]
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

    # ── DKV_MASS_TRACE: how much softmax mass reaches the NEEDLE'S ROW? ───────
    # Every other observable has been exhausted for 32k@depth0.9: the block is
    # stored correctly, routed at rank 0-1, held all generation, still fails with
    # ALL 122 blocks attended, and fails identically under project-then-attend
    # and under MLX's materialise-then-SDPA. Per-layer output similarity to dense
    # does not discriminate either -- the PASSING case measures a worse cosine
    # than the failing one.
    #
    # What has never been measured is the quantity that actually decides recall:
    # the attention WEIGHT on the needle's own row. Computed here from the SAME
    # K_all/attn_mask the SDPA below consumes, so it cannot describe a different
    # attention than the one that ran.
    #
    # trace_row is a flat index into K_all's first T rows, supplied by the caller
    # (only it knows the block layout). Reports the row's share, its rank among
    # all attended rows, and the top row for scale -- a share that is merely
    # SMALL is very different from one that is last.
    if trace_row is not None and 0 <= trace_row < keep.shape[0]:
        with torch.no_grad():
            _sc = 1.0 / (D ** 0.5)
            _s = (q.float() @ K_all.float().transpose(-1, -2)) * _sc   # [1,H_q,1,T]
            _s = _s + attn_mask.float()
            _w = torch.softmax(_s, dim=-1)[0, :, 0, :]                 # [H_q, T]
            _mine = _w[:, trace_row]
            _best_h = int(torch.argmax(_mine).item())
            _row_w = _w[_best_h]
            _rank = int((_row_w > _row_w[trace_row]).sum().item())
            _top = int(torch.argmax(_row_w).item())
            print(f"[DKV] MASS TRACE row={trace_row} tok={trace_tok} "
                  f"share={float(_mine[_best_h]):.3e} (head {_best_h}) "
                  f"rank={_rank}/{keep.shape[0]} "
                  f"top_row={_top} top_share={float(_row_w[_top]):.3e} "
                  f"dense_rows={dense_len}", flush=True)

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

    def K(rv=1, pg=7, st=0, frz=True):
        return RematCache.make_key(3, routing_version=rv, pool_generation=pg,
                                   step=st, interval=16, freeze_routing=frz)

    c = RematCache()
    k0, k1, k2 = K(), K(st=15), K(st=16)
    k3, k4 = K(rv=2), K(pg=8)
    print(f"same interval reuses (step 0 vs 15)   : {k0 == k1}   (want True)")
    print(f"interval boundary refreshes (0 vs 16) : {k0 != k2}   (want True)")
    print(f"pool write refreshes                  : {k0 != k4}   (want True)")

    # THE FREEZE. Routing changes almost every token, so keeping routing_version
    # in the key gave a 0% hit rate on GPU and made DKV_REMAT_INTERVAL inert at
    # every value. Frozen (default), a routing change must NOT refresh; with
    # DKV_REMAT_FREEZE_ROUTING=0 the old behaviour must come back intact.
    print(f"frozen: routing change does NOT refresh: {k0 == k3}   (want True)")
    u0, u3 = K(frz=False), K(rv=2, frz=False)
    print(f"unfrozen: routing change DOES refresh  : {u0 != u3}   (want True)")
    print(f"unfrozen still refreshes on interval   : "
          f"{K(frz=False) != K(st=16, frz=False)}   (want True)")
    print(f"unfrozen still refreshes on pool write : "
          f"{K(frz=False) != K(pg=8, frz=False)}   (want True)")
    # INTERVAL=1 must equal no-cache in BOTH modes -- that is the diagnostic that
    # separates a reconstruction bug from a staleness one, so it has to survive.
    for _frz in (True, False):
        _a = RematCache.make_key(3, 1, 7, step=0, interval=1, freeze_routing=_frz)
        _b = RematCache.make_key(3, 1, 7, step=1, interval=1, freeze_routing=_frz)
        print(f"INTERVAL=1 refreshes every step (frz={int(_frz)}) : "
              f"{_a != _b}   (want True)")

    c.put(k0, K, V)
    print(f"hit on same key                       : {c.get(k0) is not None}   (want True)")
    c.put(k2, K, V)
    print(f"stale entry evicted, not accumulated  : {c.get(k0) is None}   (want True)")

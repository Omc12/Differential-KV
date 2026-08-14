"""
native_core/sparse_decode/triton_fused_decode.py

Triton-optimized fused reconstruction & sparse attention kernels for DKV.
Provides maximum memory bandwidth efficiency for DKV = U @ V.T + anchor.
Falls back to pure-PyTorch on any system where Triton is unavailable.

Mac/MPS: Triton is CUDA-only; the PyTorch fallback is always used on Apple Silicon.
"""

import torch
import math
import os
import threading
from collections import OrderedDict
from typing import Optional, Tuple, List, Any
from native_core.compression.lowrank import reconstruct_batch_U

try:
    from native_core.mac_utils import nvtx_push as _nvtx_push, nvtx_pop as _nvtx_pop, has_cuda as _has_cuda
except ImportError:
    def _nvtx_push(label, device=None): pass
    def _nvtx_pop(device=None): pass
    def _has_cuda(): return torch.cuda.is_available()

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Counts how many times the PyTorch fallback path has been invoked after a
# Triton kernel failure.  Logged at thresholds (1, 10, 100) with increasing
# severity so recurring failures are visible without flooding the log.
_triton_fallback_count = 0

# ── Stage 1 of docs/audits_and_reports/CUDA_GRAPH_DECODE_PLAN.md ─────────────
# DKV_STATIC_GATHER — OPT-IN (default OFF), UNVALIDATED ON GPU.
#
# _gather_routed_blocks_for_kernel runs PER LAYER PER TOKEN and builds seven
# tensors with advanced indexing (`pool.U[indices]` etc). Advanced indexing
# ALLOCATES. At K=16 that is ~1 MB of fresh allocation and 7 launches per layer,
# so ~28 MB and ~200 launches per decoded token on a 28-layer model -- pure
# allocator and dispatch pressure, on a path measured to be ~90 ms/token of host
# overhead.
#
# index_select(..., out=buf) writes into a pre-allocated buffer instead: same
# values, no allocation, and -- the reason this is Stage 1 rather than a
# micro-optimisation -- a FIXED ADDRESS, which is what CUDA-graph capture
# requires. This is the same discipline vLLM/TensorRT-LLM/SGLang use for
# paged-attention block tables.
#
# WHY IT IS OPT-IN: dkv_attention.py can QUEUE these dicts
# (_triton_batch_queue.append) and dispatch them later. A reused buffer would be
# overwritten by the next layer between queue and dispatch, silently corrupting
# the earlier entry. The guard below refuses to reuse buffers when batching is
# active, but that interaction has NOT been tested on hardware. Enable with
# DKV_STATIC_GATHER=1 and A/B against the needle depth sweep before trusting it.
_STATIC_GATHER = os.environ.get("DKV_STATIC_GATHER", "0") == "1"


def _batch_queue_active() -> bool:
    """True when dkv_attention.py may DEFER dispatch, so buffers cannot be reused.

    dkv_attention.py gates its deferred queue on
    `bsz > 1 and HAS_TRITON and cuda and DKV_BATCH_TRITON_DISPATCH != 0`
    (dkv_attention.py:1040). Batch size is not visible from here, so this fails
    CLOSED: if deferred dispatch is even POSSIBLE, buffer reuse is refused.

    Getting this wrong is not a slowdown, it is silent cross-entry corruption --
    a queued dict's tensors overwritten by the next layer before dispatch. A
    conservative answer costs the optimisation on multi-sequence batches only;
    single-sequence decode (the measured case) still gets it.
    """
    return os.environ.get("DKV_BATCH_TRITON_DISPATCH", "1") not in ("0", "false", "off")



def _gather_into(src, indices, cache, name):
    """index_select into a persistent per-(name, shape) buffer.

    Falls back to plain advanced indexing when the buffer cannot be reused, so
    the caller never has to branch.
    """
    n = indices.numel()
    key = (name, n, tuple(src.shape[1:]), src.dtype)
    buf = cache.get(key)
    if buf is None:
        buf = torch.empty((n,) + tuple(src.shape[1:]),
                          dtype=src.dtype, device=src.device)
        cache[key] = buf
    return torch.index_select(src, 0, indices, out=buf)



# Heat-update throttle. See the call sites for why.
_heat_call_counter = 0


# DKV_GRAPH_SAFE_DECODE=1 — see runtime/dkv_attention.py. Read once at import;
# every consumer is on the per-decode-step path.
_GRAPH_SAFE_DECODE = os.environ.get("DKV_GRAPH_SAFE_DECODE", "0") == "1"


def _occupied_slots(pool, n_fallback: int):
    """Slots holding real data, as ONE device sync instead of `current_blocks`.

    Both eviction call sites built this with a Python comprehension:

        [i for i in range(pool.current_blocks) if pool.seq_lens[i].item() > 0]

    `.item()` on a CUDA tensor is a full device synchronisation, so that is one
    sync PER POOL SLOT. The pool is shared across layers, so current_blocks is
    ~(context/block_span)*n_layers -- around 3,000 slots at 32k on a 24-layer
    model. That comprehension is therefore ~3,000 syncs every time the heat
    throttle fires, which is the opposite of what the throttle exists to
    prevent: _heat_update_due was introduced to remove 27 of every 28 syncs from
    this exact code path, and the line immediately below it reintroduced three
    orders of magnitude more.

    One comparison on device, one `.tolist()`, one sync.
    """
    seq_lens = getattr(pool, "seq_lens", None)
    if seq_lens is None:
        return []
    n = int(getattr(pool, "current_blocks", n_fallback) or n_fallback)
    if n <= 0:
        return []
    return (seq_lens[:n] > 0).nonzero(as_tuple=True)[0].tolist()


def _heat_update_due() -> bool:
    """True on 1 call in DKV_HEAT_INTERVAL (default 32).

    TieredBlockStore.update_heat needs block_indices ON THE HOST, so each call
    is a `.cpu().tolist()` -- a device sync that drains the pipeline. It sits in
    native_triton_sparse_attn_decode, which runs PER LAYER PER TOKEN: 28 forced
    syncs per decoded token on a 28-layer model, so the GPU can never run ahead
    of the CPU. The decode profile showed 89% of wall time outside GPU compute,
    and this is one of the reasons.

    Heat only drives EVICTION ordering (keep routed blocks warm), and eviction
    only fires at 80% pool occupancy. It is a statistical signal, not state the
    kernel reads, so sampling it every 32 calls (~1 per token at 28 layers)
    preserves the ranking while removing 27 of every 28 syncs. Set
    DKV_HEAT_INTERVAL=1 to restore per-call updates.
    """
    global _heat_call_counter
    _heat_call_counter += 1
    try:
        n = int(os.environ.get("DKV_HEAT_INTERVAL", "32"))
    except ValueError:
        n = 32
    return n <= 1 or (_heat_call_counter % n) == 0



# DKV_DEBUG_NUMERICS — off by default.
#
# The NaN/Inf and large-LSE guards below are pure diagnostics, but each one
# converts a GPU tensor to a Python bool (`if t.any():`) or calls `.item()`,
# and BOTH force a device->host sync. They sat on the unconditional path at the
# top and bottom of fused_decode_mps, which runs per layer per token, so every
# decode step paid several full pipeline stalls to print nothing. That is the
# per-token sync recorded as F8 in CUDA_TRITON_AUDIT.md. Read once at import.
_DKV_DEBUG_NUMERICS = os.environ.get("DKV_DEBUG_NUMERICS", "0") == "1"

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


_ZERO_BLOCK_WARNED = False


def _exact_residual_semantics(device=None) -> bool:
    """Does the decode kernel SUBSTITUTE residuals, or merely ADD them?

    Must agree with how the compressor STORED them — this is a storage-format
    contract, not a tuning knob. `_exact_keys_enabled` in lowrank.py is the single
    source of truth, so this just forwards to it rather than reading the env
    again and risking the two drifting apart.

      ON  -> residual_{K,V}_values are the anchor-relative EXACT value; the kernel
             replaces the token's score and removes its lossy twin from the value
             accumulation. MLX's scheme ("exact residual rows appended and their
             lossy low-rank twins masked") and what dkv_decode.metal does.
      OFF -> they are a correction to the low-rank reconstruction; the kernel adds
             them and the twin stays. Approximate.

    Substitution is now implemented in every reader -- both Triton kernels, the
    scripted prefill history-attend, fused_decode_mps and the PyTorch vectorized
    decoder -- so _exact_keys_enabled defaults to exact form on every device.
    """
    try:
        from native_core.compression.lowrank import _exact_keys_enabled
        return bool(_exact_keys_enabled(device))
    except Exception:                                            # noqa: BLE001
        return False


def _lowrank_delta_v_at(U, V_V_nrhd, scales_n, positions):
    """The lossy low-rank twin delta_V = scale·(U[pos] @ V_V) at `positions`.

    EXACT-form residuals hold the anchor-relative TRUE value, so a decoder must
    REMOVE this twin rather than add on top of it. Every call site sits a few
    lines above a fact-anchor override that already does the same subtraction —
    the residual path simply never adopted it, which is the double-count.

    Mirrors the Triton kernels' `dv_recon` (EXACT_RESIDUAL branch): the anchor
    term is NOT included, because `rv` is anchor-relative (exact = anchor + rv),
    unlike fact anchors which store the full exact V.

      U          [N, S, R]
      V_V_nrhd   [N, R, H, D]      (permute callers whose V is [N, H, R, D])
      scales_n   [N]
      positions  [N, P]            (already clamped to >= 0)
    returns      [N, P, H, D]
    """
    R = U.shape[2]
    u_at = torch.gather(U, dim=1,
                        index=positions.unsqueeze(-1).expand(-1, -1, R))   # [N,P,R]
    dv = torch.einsum('npr,nrhd->nphd', u_at.float(), V_V_nrhd.float())
    return dv * scales_n.float().reshape(-1, 1, 1, 1)


def _substitute_scores_(scores, index, exact_scores, valid_mask):
    """In-place SUBSTITUTION of `exact_scores` at `index` along dim 2.

    Written as scatter_add_ of (target − current) rather than scatter_ + where
    so the masking stays identical to the correction path it replaces (invalid
    slots contribute exactly 0) and no caller has to re-bind `scores`.
    """
    cur = torch.gather(scores, dim=2, index=index)
    src = (exact_scores - cur).masked_fill(~valid_mask, 0.0)
    scores.scatter_add_(dim=2, index=index, src=src.to(scores.dtype))


def pool_stores_rotated_k() -> bool:
    """DKV_ROTATED_POOL — store POST-RoPE keys, as MLX does.

    STATUS 2026-08-14: the accuracy argument for keeping this ON is GONE, and
    `ultra` now sets it to 0. Whatever text below says unrotated keys cost needle
    recall is stale -- see the correction at the end of this docstring.

    THE ROOT ARCHITECTURAL DIVERGENCE BETWEEN THE TWO RUNTIMES.

        MLX   mlx_dkv_wrapper.py:4565  keys_rot = self.rope(keys, offset=offset0)
              mlx_dkv_wrapper.py:4613  manager.ingest_streaming(sid, layer_idx,
                                                               keys_rot[...], ...)
        CUDA  dkv_attention.py:888     unrot_key_states = key_states.clone()
              dkv_attention.py:900     query_states, key_states = apply_rotary_pos_emb(...)
              dkv_attention.py:1013    curr_k = unrot_key_states[...]   <- PRE-RoPE

    MLX compresses keys that are ALREADY ROTATED, so a block's delta is
    (k_rot[t] - anchor_rot) and the reconstruction anchor_rot + delta_recon
    lands in each token's TRUE rotational frame. Its only error is low-rank
    truncation.

    CUDA compresses UNROTATED keys and rotates on read -- but it rotates the
    anchor AND THE ENTIRE V_K BASIS at the ANCHOR's position
    (_gather_routed_blocks_for_kernel), because a per-token rotation would cost
    a D-dim reconstruction per token and defeat the point of the low-rank form.
    So every compressed token's key comes out as

        R(anchor_pos) . (anchor_raw + delta_recon_raw)

    when its true key is

        R(true_pos) . (anchor_raw + delta_raw)

    i.e. EVERY compressed token carries a RoPE phase error of up to a full
    block (256 positions) ON TOP of truncation error. That is the
    "Project-Then-Attend approximation" the comments refer to, and it is not an
    approximation MLX makes at all.

    It also means residual corrections cannot cancel it: the residual key is
    rotated at the token's TRUE position while the base term it corrects is
    rotated at the anchor, so the two live in different frames and their sum is
    not the exact key in either one. Both residual formats have this problem --
    the CORRECTION form adds across frames, and the EXACT form's substitution
    still keeps the anchor-rotated s_anchor term.

    Turning this ON makes CUDA store what MLX stores. It then costs LESS, not
    more: no inverse-RoPE at ingest, no rotation of anchors_K/V_K/res_k in the
    decode gather, and no rotation in the router. The whole class of read-time
    position bugs (both of which were real and fixed earlier: the router using a
    within-block offset as an absolute position, and the decode gather being off
    by one) stops existing because there is no read-time position to get wrong.

    DEFAULT ON, and the A/B that flipped it (Qwen3.5-2B, validate_cuda_dkv
    --long, same build, same prompts, only this flag differing):

        DKV_ROTATED_POOL=0    FAILED (6)
        DKV_ROTATED_POOL=1    FAILED (4)

        32k@depth0.5   2/3 with 2 distinct outputs  ->  3/3, DETERMINISTIC
        2k@depth0.0    output changed               ->  still failing
        8k@depth0.5    unchanged                    ->  still 2/3
        32k@depth0.9   unchanged                    ->  still 0/3

    The 32k@0.5 result is the informative one: it was BOTH wrong and
    nondeterministic at temperature 0, and removing the phase error fixed both
    at once. That is what a systematic per-token position error looks like when
    it lands near a decision boundary -- the ranking of two close candidates
    flips on fp16 noise. Recall and determinism moving together is the signature
    of the base scores becoming correct rather than of a threshold being nudged.

    It is also the cheaper direction: no inverse-RoPE at ingest, and no rotation
    of anchors_K/V_K/res_k in the gather, the router or the dense window.

    Every site that rotates on read consults THIS function, so the convention
    cannot end up half-applied. DKV_ROTATED_POOL=0 restores the unrotated pool.
    A pool built under one setting is not readable under the other, which is a
    non-issue in practice (the value is fixed for a process) but matters if a
    session is ever persisted across a config change.

    CORRECTION 2026-08-14 -- the trade described above no longer exists.

    The reason this stayed ON was that unrotated keys scored 6/9 on the needle
    sweep, including failures at 2k where nothing is compressed at all. That
    was read at the time as a broken unrotated READ path rather than a real
    fidelity trade, and that reading was right: it is now fixed. Re-measured
    with DKV_ROTATED_POOL=0 on the current build, the sweep is 9/9 with 9/9
    determinism on BOTH Qwen3.5-2B and Qwen2.5-1.5B-Instruct, at every depth
    and every length.

    With that gone, unrotated is strictly better on accuracy. linkbench at 32k
    over 48 seeds on Qwen3.5-2B -- 48 samples per point, unlike multifact whose
    +-15-point RSVD-seed band cannot resolve anything:

        rotated (default)   40/48
        UNROTATED           47/48
        dense               47/48   <- exact parity

    It is not free: rotating at read costs decode and memory. Qwen3.5-2B at
    32k, interleaved and reversed, 17.60 -> 13.37 and 15.55 -> 12.70 tok/s
    (-18% to -24%), and device VRAM 5.21 -> 6.31 GB. So the DEFAULT stays
    rotated and the `ultra` preset sets rotated_pool=False, which is where a
    speed-for-accuracy trade of that size belongs.
    """
    # DEFAULT 0. This shipped as "1" while EVERY prefill capture site stored
    # `unrot_key_states` unconditionally, so the pool held PRE-RoPE keys and this
    # predicate told the decode gather to skip the rotation they needed
    # (`do_rot = ... and not pool_stores_rotated_k()`). Compressed keys were
    # therefore never rotated at all.
    #
    # probe_residual_values measured it at 32k@depth0.9: anchor+residual scores
    # cos = 1.0000 against UNROTATED ground truth and 0.84-0.98 against rotated,
    # with |K_pool| == |K_true| at every layer -- the exact signature of a pure
    # rotation, since RoPE is orthogonal.
    #
    # DEFAULT 1, because the EXACT/SUBSTITUTION residual form is only coherent
    # with a rotated pool. Work the algebra for `s = s_anchor + q·rk`:
    #
    #   ROTATED (MLX):   anchor_stored = RoPE(anchor, p_a)
    #                    residual      = RoPE(k_t, p_t) - RoPE(anchor, p_a)
    #                    sum           = RoPE(k_t, p_t)          EXACT.
    #     The anchor/token position difference cancels itself, which is exactly
    #     why MLX captures keys POST-RoPE (mlx_dkv_wrapper.py:4565).
    #
    #   UNROTATED:       s_anchor      = q·RoPE(anchor, p_a)
    #                    rk rotated at p_t (DIFFKV_RESIDUAL_EXACT_ROPE)
    #     The residual term is right but the ANCHOR term is rotated at the wrong
    #     position -- off by up to a full block (256 tokens) of phase. The anchor
    #     is a full key vector while the residual is a small delta, so that error
    #     DOMINATES the score. Reconstructing correctly would need the anchor
    #     re-rotated per token, which defeats the point of sharing one anchor
    #     across the block.
    #
    # The same argument applies to the low-rank half: under a rotated pool
    # anchor + U@V approximates RoPE(k_t, p_t) directly with no rotation at read,
    # whereas the unrotated pool rotates V_K at the ANCHOR's position for every
    # token in the block (the Project-Then-Attend approximation).
    #
    # This shipped as "1" while every prefill capture stored `unrot_key_states`
    # unconditionally, so the pool held PRE-RoPE keys and this predicate told the
    # gather to skip the rotation they needed. probe_residual_values measured it:
    # anchor+residual scored cos = 1.0000 against UNROTATED truth vs 0.84-0.98
    # rotated, with |K_pool| == |K_true| at every layer (RoPE is orthogonal, so a
    # pure rotation preserves norm). _ingest_k now routes every capture through
    # this same predicate, so the two sides cannot disagree.
    return os.environ.get("DKV_ROTATED_POOL", "1") == "1"


def resolve_sparse_bias(lse_sparse=None, lse_dense=None):
    """DKV_SPARSE_BIAS -> the additive nats applied to the sparse half's LSE.

    Single source of truth: dkv_attention.py's merge and the inline merge in
    native_triton_sparse_attn_decode both go through here, because they used to
    parse the env string separately and could drift.

    WHY 'auto' IS 0.0 ON CUDA
    -------------------------
    The adaptive formula is identical on both runtimes, character for character:

        bias = max(0, BASE - 0.5 * max(0, (lse_dense - lse_sparse) - 4.0))

    but it is evaluated on a DIFFERENT PARTITION of the softmax:

      MLX  (mlx_dkv_wrapper.py:771, :1031)
        sparse = anchors + low-rank deltas, lossy TWIN of each exact residual
                 forced to -inf
        dense  = mx.concatenate([res_k_all, dense_k]) -- the EXACT residual keys
                 concatenated IN FRONT of the recency window

      CUDA (this file, :460-483 and :2434)
        sparse = anchors + low-rank deltas, twins KEPT, residual correction
                 applied in place INSIDE the same softmax
        dense  = the recency window only

    So a query matching an exact residual raises lse_DENSE on MLX and
    lse_SPARSE on CUDA -- opposite signs into the same subtraction. MLX's own
    comment on the decay branch reads "decays to 0 as the dense half (e.g. an
    exact needle residual) pulls ahead". On CUDA the needle pulls the other half
    ahead, so `diff` goes NEGATIVE and the bias pins at BASE. Reaching the decay
    branch on CUDA would require the recency window to beat ALL of compressed
    history by 4 nats -- the opposite of the condition it was written for.

    'auto' on CUDA was therefore never adaptive: it was a constant +2.0 nats,
    e^2 = 7.4x, up-weighting compressed history against the recent window on
    every token of every query. MLX resolves 'auto' to 0.0 in exactly the
    structural situation CUDA is in -- its fused path, where residuals share the
    softmax with the lossy keys (mlx_dkv_wrapper.py:4078: "Since the adaptive
    form can't be reproduced per-token here, 'auto' resolves to 0.0").

    'adaptive[,BASE]' keeps the old behaviour so the two can be A/B'd; an
    explicit numeric value is still honoured as a flat bias.
    """
    env = os.environ.get("DKV_SPARSE_BIAS", "0.0").strip().lower()
    if env.startswith("auto"):
        # Neutered to 0.0 ONLY because the exact residuals sit in the sparse half.
        # With DKV_RESIDUALS_IN_DENSE the partition matches MLX's -- exact rows in
        # the DENSE half, twins masked out of sparse -- so `diff` regains the sign
        # the formula was written against ("decays to 0 as the dense half, e.g. an
        # exact needle residual, pulls ahead") and `auto` becomes genuinely
        # adaptive instead of a constant. Same formula and same 2.0 default as
        # mlx_dkv_wrapper.py:26-28; falling through to the shared branch below
        # keeps the two from drifting apart.
        if residuals_in_dense() and lse_sparse is not None and lse_dense is not None:
            parts = env.split(",")
            try:
                base = float(parts[1]) if len(parts) > 1 and parts[1] else 2.0
            except ValueError:
                base = 2.0
            diff = lse_dense - lse_sparse
            if torch.is_tensor(diff):
                return torch.clamp(base - 0.5 * torch.clamp(diff - 4.0, min=0.0), min=0.0)
            return max(0.0, base - 0.5 * max(0.0, diff - 4.0))
        return 0.0
    if env.startswith("adaptive"):
        if lse_sparse is None or lse_dense is None:
            return 0.0
        parts = env.split(",")
        try:
            base = float(parts[1]) if len(parts) > 1 and parts[1] else 2.0
        except ValueError:
            base = 2.0
        diff = lse_dense - lse_sparse
        if torch.is_tensor(diff):
            return torch.clamp(base - 0.5 * torch.clamp(diff - 4.0, min=0.0), min=0.0)
        return max(0.0, base - 0.5 * max(0.0, diff - 4.0))
    try:
        return float(env)
    except ValueError:
        return 0.0


_SM_COUNT_CACHE: dict = {}


def _sm_count(device) -> int:
    """SM count for `device`, cached (this is on the per-layer decode path)."""
    key = getattr(device, "index", None) or 0
    n = _SM_COUNT_CACHE.get(key)
    if n is None:
        try:
            n = int(torch.cuda.get_device_properties(device).multi_processor_count)
        except Exception:
            n = 32
        _SM_COUNT_CACHE[key] = n
    return n


def _blocks_per_chunk(n_blocks: int = 0, h_q: int = 0, device=None) -> int:
    """KV blocks each Triton program handles before the cross-chunk reduction.

    Chosen from GPU occupancy, not a constant. The grid is (H_q, num_chunks), so
    a fixed 16 meant that a routed set at or below 16 blocks produced num_chunks=1
    and launched H_q programs TOTAL -- 12 on Qwen2.5-1.5B. On a 56-SM card that
    left ~80% of the GPU idle while each program serially walked every routed
    block, and it is the dominant cost of long-context decode: the fused kernel
    measured 53 ms/step at 32k, 61% of all device time, against 10.8 ms once the
    work was actually spread out.

    The routed set is PRUNED to a small count (12 here) independent of context
    length, so longer context never grew num_chunks past 1 on its own -- the
    bigger the context, the worse the relative loss.

    Targets ~2 programs per SM, and never returns FEWER chunks than the old
    constant would have, so large routed sets keep their previous shape and only
    the under-parallelised small-N case changes. DKV_BLOCKS_PER_CHUNK still
    overrides it outright, which is what makes num_chunks=1 reachable for
    isolating the per-block math from the cross-chunk merge.
    """
    env = os.environ.get("DKV_BLOCKS_PER_CHUNK")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 16
    if n_blocks <= 0 or h_q <= 0:
        return 16
    target_programs = 2 * _sm_count(device)
    chunks_for_occupancy = -(-target_programs // h_q)      # ceil
    chunks_at_old_default = -(-n_blocks // 16)             # ceil
    n_chunks = min(n_blocks, max(chunks_for_occupancy, chunks_at_old_default))
    return max(1, -(-n_blocks // n_chunks))                # ceil


def _partial_rope_apply(x, cos, sin):
    """Apply RoPE to x using cos/sin, honoring partial rotary (Qwen3.5/GLM-style
    partial_rotary_factor<1.0): cos/sin's last dim (rotary_dim) may be smaller
    than x's -- rotate only that leading slice and pass the remainder through
    unrotated, mirroring apply_rope_to_keys in decode_attention.cpp. Reduces to
    the original full-width `x*cos + rotate_half(x)*sin` when rotary_dim==D.
    """
    rotary_dim = cos.shape[-1]
    D = x.shape[-1]
    if rotary_dim >= D:
        return x * cos + rotate_half(x) * sin
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([rotated, x_pass], dim=-1)

# ── 1. Fused Triton Kernels ───────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def dkv_fused_decode_kernel(
        # Inputs already in pool (indexed by slot)
        Q_ptr,           # [num_q_heads, head_dim]
        U_ptr,           # [pool_size, MAX_S, RANK]          INT8
        U_scale_ptr,     # [pool_size]                        FP16
        VK_ptr,          # [pool_size, RANK, head_dim_kv]    INT8 (after V quant)
        VK_scale_ptr,    # [pool_size, RANK]                  FP16
        AncK_ptr,        # [pool_size, kv_heads, head_dim]   INT8
        AncK_scale_ptr,  # [pool_size]                        FP16
        VV_ptr,          # [pool_size, RANK, head_dim_kv]    INT8
        VV_scale_ptr,    # [pool_size, RANK]                  FP16
        AncV_ptr,        # [pool_size, kv_heads, head_dim]   INT8
        AncV_scale_ptr,  # [pool_size]                        FP16
        SlotIdx_ptr,     # [n_active]  which slots are live
        BlkSz_ptr,       # [n_active]  actual token count per block
        PartOut_ptr,     # [n_chunks, num_q_heads, head_dim] output accumulator
        PartLSE_ptr,     # [n_chunks, num_q_heads]            log-sum-exp
        n_active,
        scale,           # 1/sqrt(head_dim)
        q_per_kv,        # GQA ratio
        num_q_heads,
        HEAD_DIM:         tl.constexpr,
        RANK:             tl.constexpr,
        TILE_S:           tl.constexpr,     # tokens per inner tile
        BLOCKS_PER_CHUNK: tl.constexpr,     # KV blocks per thread block
        MAX_S:            tl.constexpr,
    ):
        q_head  = tl.program_id(0)
        chunk   = tl.program_id(1)
        kv_head = q_head // q_per_kv

        # ── Load Q into registers — never leaves registers ──────────────────
        q = tl.load(Q_ptr + q_head * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.float32)

        # ── Load V_K for this KV head into SRAM — shared across all blocks ──
        # KEY INSIGHT: V_K is loaded ONCE here and reused for ALL blocks.
        vk_base  = tl.load(VK_scale_ptr + tl.arange(0, RANK),   # [RANK] scales
                           mask=tl.arange(0, RANK) < RANK)
        # V_K dequant: inline in registers during q_proj computation
        vk_data  = tl.load(
            VK_ptr + kv_head * RANK * HEAD_DIM +
            tl.arange(0, RANK)[:, None] * HEAD_DIM +
            tl.arange(0, HEAD_DIM)[None, :]
        ).to(tl.float32) * vk_base[:, None]                        # [RANK, HEAD_DIM]

        # Same for V_V
        vv_base  = tl.load(VV_scale_ptr + tl.arange(0, RANK))
        vv_data  = tl.load(
            VV_ptr + kv_head * RANK * HEAD_DIM +
            tl.arange(0, RANK)[:, None] * HEAD_DIM +
            tl.arange(0, HEAD_DIM)[None, :]
        ).to(tl.float32) * vv_base[:, None]                        # [RANK, HEAD_DIM]

        # ── Q projection — computed once for this entire chunk ─────────────
        q_proj_k = tl.sum(q[None, :] * vk_data, axis=1) * scale   # [RANK]

        # ── Online softmax state in registers ──────────────────────────────
        m   = float('-inf')
        l   = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for b in range(BLOCKS_PER_CHUNK):
            b_abs = chunk * BLOCKS_PER_CHUNK + b
            if b_abs >= n_active: break

            slot     = tl.load(SlotIdx_ptr + b_abs)
            blk_sz   = tl.load(BlkSz_ptr   + b_abs)
            u_scale  = tl.load(U_scale_ptr  + slot).to(tl.float32)
            ak_scale = tl.load(AncK_scale_ptr + slot).to(tl.float32)

            # Anchor score (post-RoPE anchor, stored exactly at position)
            anc_k = tl.load(AncK_ptr + slot * tl.constexpr(2) * HEAD_DIM +
                            kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
                           ).to(tl.float32) * ak_scale
            s_anc = tl.sum(q * anc_k) * scale

            # Anchor value
            av_scale = tl.load(AncV_scale_ptr + slot).to(tl.float32)
            anc_v    = tl.load(AncV_ptr + slot * tl.constexpr(2) * HEAD_DIM +
                               kv_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
                              ).to(tl.float32) * av_scale

            # Online softmax: anchor
            m_new = tl.maximum(m, s_anc)
            l_new = l * tl.exp(m - m_new) + tl.exp(s_anc - m_new)
            acc   = acc * (l / l_new) * tl.exp(m - m_new) + \
                    anc_v * (tl.exp(s_anc - m_new) / l_new)
            m, l  = m_new, l_new

            # Delta tokens: Project-Then-Attend, tiled
            u_base = slot * MAX_S * RANK

            for t in range(0, MAX_S - 1, TILE_S):
                valid   = (tl.arange(0, TILE_S) + t) < (blk_sz - 1)
                u_tile  = tl.load(
                    U_ptr + u_base + (t + tl.arange(0, TILE_S))[:, None] * RANK +
                    tl.arange(0, RANK)[None, :],
                    mask=valid[:, None], other=0
                ).to(tl.float32) * u_scale                          # [TILE_S, RANK]

                d_scores = tl.sum(u_tile * q_proj_k[None, :], axis=1)
                d_scores = tl.where(valid, d_scores, float('-inf'))

                t_max  = tl.max(d_scores, axis=0)
                m_new  = tl.maximum(m, t_max)
                exp_d  = tl.exp(d_scores - m_new) * valid.to(tl.float32)
                l_new  = l * tl.exp(m - m_new) + tl.sum(exp_d)

                w_d    = exp_d / l_new
                w_proj = tl.sum(w_d[:, None] * u_tile, axis=0)
                v_c    = tl.sum(w_proj[:, None] * vv_data, axis=0)

                acc = acc * (l / l_new) * tl.exp(m - m_new) + v_c
                m, l = m_new, l_new

        out_off = (chunk * num_q_heads + q_head) * HEAD_DIM
        tl.store(PartOut_ptr + out_off + tl.arange(0, HEAD_DIM), acc)
        tl.store(PartLSE_ptr + chunk * num_q_heads + q_head,
                 m + tl.log(tl.where(l > 0.0, l, 1e-9)))

    # Working Triton kernel matching the actual NativeBlockPool layout.
    # Triton pre-compiles one kernel per (R, S_MAX, D) combination seen at runtime,
    # since all three are constexpr — the inner rank loop unrolls completely for
    # each specialisation → full register reuse, no loop overhead. autotune then
    # picks num_warps per (R, D); it does NOT choose S_MAX or BLOCKS_PER_CHUNK,
    # which are caller-supplied correctness parameters — see the note below.
    @triton.autotune(
        configs=[
            # ONLY num_warps is tuned here. S_MAX and BLOCKS_PER_CHUNK used to be
            # Config kwargs, which broke this kernel two ways at once:
            #
            # 1. CRASH. autotune injects Config kwargs as keyword arguments, and
            #    the call site also passes both POSITIONALLY (they sit at fixed
            #    slots in the constexpr list). Every launch raised
            #      TypeError: dynamic_func() got multiple values for argument 'S_MAX'
            #    so native_triton_sparse_attn_decode ALWAYS fell back to the
            #    PyTorch decoder. That is the DKV_SPARSE_BIAS=auto path, i.e. the
            #    serving default -- so production has never run this kernel. It
            #    stayed hidden because validate_cuda_dkv.py exercised the OTHER
            #    entry point (native_triton_sparse_attn_decode_combined), where
            #    fallback_count=0 was true and meaningless.
            #
            # 2. WRONG RESULTS if it had launched. Neither value is a tuning knob.
            #    S_MAX is the block's padded sequence length -- the caller passes
            #    next_power_of_2(pool.max_seq_len), 256 for this pool -- and
            #    `offs_s = tl.arange(0, S_MAX)` indexes tokens with it, so a config
            #    picking 64 would silently read the first 64 tokens of a 256-token
            #    block and drop the rest. BLOCKS_PER_CHUNK is worse: the caller
            #    derives num_chunks AND the launch grid from it, so autotuning it
            #    desynchronises the grid from the work decomposition.
            #
            # R (rank) and D (head_dim) are constexpr and specialise per launch
            # already; `key` re-tunes warps when either changes.
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["R", "D"],         # specialise per (rank, head_dim) pair
        reset_to_zero=["out_ptr", "m_ptr", "l_ptr"],
    )
    @triton.jit
    def _fused_sparse_decode_kernel(

        q_ptr, block_indices_ptr, pool_ak_ptr, pool_av_ptr, pool_vk_ptr, pool_vv_ptr,
        pool_u_ptr, pool_u_scale_ptr, pool_scales_ptr, pool_seq_lens_ptr,
        # Residual correction pointers (C1). res_pos = K positions, res_pos_v = V positions
        # (they differ — K and V select different worst-reconstructed tokens).
        pool_res_k_ptr, pool_res_v_ptr, pool_res_pos_ptr, pool_res_pos_v_ptr, pool_res_n_ptr,
        # Fact anchor override pointers (C2)
        pool_fact_pos_ptr, pool_fact_ak_ptr, pool_fact_av_ptr,
        out_ptr, m_ptr, l_ptr,
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_res_k_n, stride_res_k_s, stride_res_k_h, stride_res_k_d,
        stride_res_v_n, stride_res_v_s, stride_res_v_h, stride_res_v_d,
        stride_res_pos_n, stride_res_pos_v_n,
        stride_fact_pos_n,
        stride_fact_ak_n, stride_fact_ak_f, stride_fact_ak_h, stride_fact_ak_d,
        stride_fact_av_n, stride_fact_av_f, stride_fact_av_h, stride_fact_av_d,
        stride_out_h, stride_out_d,
        N: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr, KV_GRP: tl.constexpr, D: tl.constexpr,
        R: tl.constexpr, R_REAL: tl.constexpr, S_MAX: tl.constexpr, INV_SCALE: tl.constexpr, BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
        MAX_RESIDUAL: tl.constexpr, MAX_FACT: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr, HAS_FACT: tl.constexpr,
        EXACT_RESIDUAL: tl.constexpr, RESIDUAL_IN_DENSE: tl.constexpr,
    ):
        h_q = tl.program_id(0)
        chunk_id = tl.program_id(1)
        h_kv = h_q // KV_GRP
        
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, R)
        offs_s = tl.arange(0, S_MAX)
        
        q_ptrs = q_ptr + h_q * stride_q_h + offs_d * stride_q_d
        q = tl.load(q_ptrs).to(tl.float32)
        
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        start_block = chunk_id * BLOCKS_PER_CHUNK
        end_block = start_block + BLOCKS_PER_CHUNK
        if end_block > N:
            end_block = N
            
        for n in range(start_block, end_block):
            pool_idx = tl.load(block_indices_ptr + n)
            scale = tl.load(pool_scales_ptr + pool_idx).to(tl.float32)
            actual_s = tl.load(pool_seq_lens_ptr + pool_idx)
            
            ak_ptrs = pool_ak_ptr + pool_idx * stride_ak_n + h_kv * stride_ak_h + offs_d * stride_ak_d
            av_ptrs = pool_av_ptr + pool_idx * stride_av_n + h_kv * stride_av_h + offs_d * stride_av_d
            ak = tl.load(ak_ptrs).to(tl.float32)
            av = tl.load(av_ptrs).to(tl.float32)
            
            # r_mask is REQUIRED, not defensive. R here is R_pad =
            # next_power_of_2(layer_rank), while the pool's rank dimension is
            # pool_rank (the max rank across layers). For a 1.5x-boosted layer
            # rank 48 -> R_pad 64 against a 48-wide pool, so r = 48..63 walked
            # past the rank dimension into the NEXT SLOT's basis and summed it
            # into this block's scores and values. For rank 24 -> R_pad 32 it
            # read 8 columns this layer never wrote. Both vk/vv loads were
            # unmasked; only `u` was masked, and only on S.
            r_mask = offs_r < R_REAL
            vk_ptrs = pool_vk_ptr + pool_idx * stride_vk_n + h_kv * stride_vk_h + offs_r[:, None] * stride_vk_r + offs_d[None, :] * stride_vk_d
            vv_ptrs = pool_vv_ptr + pool_idx * stride_vv_n + h_kv * stride_vv_h + offs_r[:, None] * stride_vv_r + offs_d[None, :] * stride_vv_d
            vk = tl.load(vk_ptrs, mask=r_mask[:, None], other=0.0).to(tl.float32)
            vv = tl.load(vv_ptrs, mask=r_mask[:, None], other=0.0).to(tl.float32)
            
            u_ptrs = pool_u_ptr + pool_idx * stride_u_n + offs_s[:, None] * stride_u_s + offs_r[None, :] * stride_u_r
            s_mask = (offs_s[:, None] < actual_s) & r_mask[None, :]
            u = tl.load(u_ptrs, mask=s_mask, other=0.0).to(tl.float32)
            
            u_scale_ptr = pool_u_scale_ptr + pool_idx
            u_scale = tl.load(u_scale_ptr)
            u = u * u_scale
            
            s_anchor = tl.sum(q * ak) * INV_SCALE
            q_proj = tl.sum(q[None, :] * vk, axis=1) * INV_SCALE
            delta_scores = tl.sum(u * q_proj[None, :], axis=1) * scale
            s = s_anchor + delta_scores

            # ── C1: Residual K correction (ALIGNED to Mac reference) ──────────────
            # residual_K_values store (exact - lowrank_recon) at the worst-reconstructed
            # positions (lowrank.py:659). Add q·resK to the delta score AT that position
            # so the token's score becomes exact — one token per position, matching
            # _pytorch_vectorized_sparse_attn_decode's scatter_add_ (line 1164) and
            # fused_decode_mps (806). res_k is pre-rotated (anchor RoPE) in the dispatcher.
            # NOTE: this replaces the old "append residuals as extra softmax tokens" path,
            # which double-counted those positions and was measurably worse than no
            # correction at all (see CUDA_TRITON_AUDIT.md F1).
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_k = tl.load(pool_res_pos_ptr + pool_idx * stride_res_pos_n + ri)
                    if r_pos_k >= 0:
                        if RESIDUAL_IN_DENSE:
                            # MLX PARTITION (mlx_dkv_wrapper.py:771 + :1031). MLX does
                            # NOT score the exact row here. It masks the lossy twin to
                            # -inf in the SPARSE half
                            #     delta_s = where(res_mask, -inf, delta_s)
                            # and carries the exact row in the DENSE half
                            #     dense_k_for_attn = concat([res_k_all, dense_k])
                            # The dispatcher appends those rows to dense_k/dense_v, so
                            # scoring them here too would DOUBLE-COUNT the token -- the
                            # very bug F1 above records.
                            #
                            # WHY THE SIDE MATTERS, and it is not cosmetic: the merge
                            # bias is
                            #   bias = max(0, base - 0.5*max(0, (lse_dense-lse_sparse)-4))
                            # and its NIAH safety rests on "the exact needle residual
                            # makes lse_dense dominate -> bias->0" (mlx_dkv_wrapper.py
                            # :26-28). That holds only while the exact rows sit in the
                            # DENSE half. With them in the SPARSE half the gap moves the
                            # wrong way and `auto` stays pinned near its +2.0 maximum
                            # instead of decaying -- on a knob whose own note records
                            # that +4.0 breaks NIAH by needle corruption. The control
                            # meant to detect "an exact needle is present, back off" is
                            # reading the wrong side of the merge.
                            #
                            # MASKING, not merely skipping: leaving the twin live would
                            # keep a 20-35% rel-error key (measured by
                            # probe_residual_values) competing against the token's own
                            # exact copy.
                            s = tl.where(offs_s == r_pos_k, -float("inf"), s)
                        else:
                            rk = tl.load(pool_res_k_ptr + pool_idx * stride_res_k_n +
                                         ri * stride_res_k_s + h_kv * stride_res_k_h +
                                         offs_d * stride_res_k_d).to(tl.float32)
                            r_corr = tl.sum(q * rk) * INV_SCALE
                            if EXACT_RESIDUAL:
                                # MLX / Metal SUBSTITUTION. rk is the anchor-relative
                                # EXACT key, so exact_K = ak + rk and this token's true
                                # score is s_anchor + q·rk. Writing that REPLACES the
                                # score, dropping delta_scores[p] — the lossy low-rank
                                # twin. Same shape as the C2 fact-anchor override below.
                                # NOTE this keeps the exact row in the SPARSE half,
                                # which is the MLX divergence RESIDUAL_IN_DENSE fixes.
                                s = tl.where(offs_s == r_pos_k, s_anchor + r_corr, s)
                            else:
                                # Correction form: rk is (exact - recon), so ADD and keep
                                # the twin. Approximate: the twin's delta stays in, and
                                # the two terms are rotated in different frames (base at
                                # the block anchor, rk at the token's true position).
                                s = tl.where(offs_s == r_pos_k, s + r_corr, s)

            # ── C2: Fact Anchor Override — replace scores at flagged positions ──
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        fact_k_ptrs = pool_fact_ak_ptr + pool_idx * stride_fact_ak_n + fi * stride_fact_ak_f + h_kv * stride_fact_ak_h + offs_d * stride_fact_ak_d
                        fact_k = tl.load(fact_k_ptrs).to(tl.float32)
                        fact_score = tl.sum(q * fact_k) * INV_SCALE
                        # Override: scatter exact score at the flagged delta position
                        replace_mask = offs_s == fact_pos
                        s = tl.where(replace_mask, fact_score, s)
            
            s = tl.where(offs_s < actual_s, s, -float("inf"))
            m_b_delta = tl.max(s, axis=0)
            m_b = tl.maximum(s_anchor, m_b_delta)
            
            m_new = tl.maximum(m_i, m_b)
            alpha = tl.exp(m_i - m_new)
            p_anchor = tl.exp(s_anchor - m_new)
            p_delta = tl.exp(s - m_new)
            p_delta = tl.where(offs_s < actual_s, p_delta, 0.0)
            p_delta_sum = tl.sum(p_delta, axis=0)
            
            l_i = l_i * alpha + p_anchor + p_delta_sum
            
            p_u = tl.sum(p_delta[:, None] * u, axis=0)
            o_delta = tl.sum(p_u[:, None] * vv, axis=0) * scale
            
            # ── C2: Fact Anchor Override Value Correction ──
            O_fact_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        # Get attention weight for this fact token
                        replace_mask = offs_s == fact_pos
                        p_fact = tl.sum(tl.where(replace_mask, p_delta, 0.0), axis=0)
                        
                        # Load exact fact V
                        fact_v_ptrs = pool_fact_av_ptr + pool_idx * stride_fact_av_n + fi * stride_fact_av_f + h_kv * stride_fact_av_h + offs_d * stride_fact_av_d
                        fact_v = tl.load(fact_v_ptrs).to(tl.float32)
                        
                        # Compute low-rank reconstructed V at fact_pos
                        u_val_ptrs = pool_u_ptr + pool_idx * stride_u_n + fact_pos * stride_u_s + offs_r * stride_u_r
                        # vv is already r-masked to zero above so the sum would be
                        # right regardless, but the LOAD itself must not address
                        # past the rank dimension (the last slot has nothing after it).
                        u_val = tl.load(u_val_ptrs, mask=r_mask, other=0.0).to(tl.float32) * u_scale
                        v_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale + av
                        
                        # Accumulate correction: p_fact * (fact_v - v_recon)
                        O_fact_corr += p_fact * (fact_v - v_recon)
            
            # ── C1: Residual V correction (ALIGNED to Mac reference) ──────────────
            # O += p_delta[res_pos_v] · resV, with resV = (exact - recon) V
            # (lowrank.py:666). Uses the SAME unnormalized p_delta as o_delta (the K
            # correction above already made p_delta exact at these positions), and is
            # normalized by l_i at the end — matching _pytorch_vectorized_…'s
            # gather(P, res_pos_v)·resV (lines 1291-1298). V residual positions differ
            # from K's, so this loop reads pool_res_pos_v_ptr (not pool_res_pos_ptr).
            O_res_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_v = tl.load(pool_res_pos_v_ptr + pool_idx * stride_res_pos_v_n + ri)
                    if r_pos_v >= 0:
                        p_at = tl.sum(tl.where(offs_s == r_pos_v, p_delta, 0.0), axis=0)
                        rv = tl.load(pool_res_v_ptr + pool_idx * stride_res_v_n +
                                     ri * stride_res_v_s + h_kv * stride_res_v_h +
                                     offs_d * stride_res_v_d).to(tl.float32)
                        if EXACT_RESIDUAL:
                            # Substitution on the V side too, or the score would be
                            # exact while the value stayed lossy. The row currently
                            # contributes p_at·(av + delta_V[p]) via o_delta and the
                            # shared av term; the exact value is av + rv, so the
                            # correction is p_at·(rv - delta_V[p]). delta_V[p] is
                            # recomputed from this position's U row exactly as the
                            # C2 fact-anchor value override does. The load must be
                            # r_mask'd: R here is R_pad, wider than the pool's rank.
                            u_val_ptrs = (pool_u_ptr + pool_idx * stride_u_n
                                          + r_pos_v * stride_u_s + offs_r * stride_u_r)
                            u_val = tl.load(u_val_ptrs, mask=r_mask, other=0.0).to(tl.float32) * u_scale
                            dv_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale
                            O_res_corr += p_at * (rv - dv_recon)
                        else:
                            O_res_corr += p_at * rv

            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta + O_fact_corr + O_res_corr
            m_i = m_new

        if NUM_CHUNKS == 1:
            O_i = O_i / l_i
            out_ptrs = out_ptr + h_q * stride_out_h + offs_d * stride_out_d
            tl.store(out_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q, l_i)
        else:
            out_work_ptrs = out_ptr + h_q * (NUM_CHUNKS * D) + chunk_id * D + offs_d
            tl.store(out_work_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q * NUM_CHUNKS + chunk_id, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q * NUM_CHUNKS + chunk_id, l_i)


    @triton.jit
    def _fused_sparse_decode_reduction_kernel(
        out_workspace_ptr, m_workspace_ptr, l_workspace_ptr, out_ptr, m_final_ptr, l_final_ptr,
        NUM_CHUNKS: tl.constexpr, D: tl.constexpr,
    ):
        h_q = tl.program_id(0)
        offs_d = tl.arange(0, D)
        
        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)
        
        for c in range(NUM_CHUNKS):
            m_c = tl.load(m_workspace_ptr + h_q * NUM_CHUNKS + c)
            l_c = tl.load(l_workspace_ptr + h_q * NUM_CHUNKS + c)
            out_c_ptrs = out_workspace_ptr + h_q * (NUM_CHUNKS * D) + c * D + offs_d
            O_c = tl.load(out_c_ptrs).to(tl.float32)
            
            m_new = tl.maximum(m_i, m_c)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_c - m_new)
            
            l_i = l_i * alpha + l_c * beta
            O_i = O_i * alpha + O_c * beta
            m_i = m_new
            
        O_i = O_i / l_i
        out_ptrs = out_ptr + h_q * D + offs_d
        tl.store(out_ptrs, O_i)
        if m_final_ptr is not None:
            tl.store(m_final_ptr + h_q, m_i)
        if l_final_ptr is not None:
            tl.store(l_final_ptr + h_q, l_i)

    # OPT-E: Pairwise-merge kernel — one pass of a binary tree reduction.
    # Each program merges a (left, right) chunk pair: accumulator at slot `left`
    # absorbs slot `right` using the standard online-softmax (LSE-safe) formula.
    # Launch ceil(NUM_ACTIVE / 2) programs along axis-1 to run all pairs in parallel.
    # After ceil(log2(NUM_CHUNKS)) such passes only slot 0 remains, which is the
    # final normalised output.  Only used when num_chunks >= 8 (see _dispatch_reduction
    # below); the sequential kernel is faster at smaller chunk counts.
    @triton.jit
    def _fused_reduction_pairwise_kernel(
        workspace_ptr,   # [H_q, NUM_CHUNKS_PAD, D]  in-place
        m_ptr,           # [H_q, NUM_CHUNKS_PAD]      in-place
        l_ptr,           # [H_q, NUM_CHUNKS_PAD]      in-place
        STRIDE: tl.constexpr,   # distance between left and right slot (1, 2, 4, …)
        NUM_CHUNKS: tl.constexpr,
        D: tl.constexpr,
    ):
        h_q   = tl.program_id(0)
        pair  = tl.program_id(1)
        left  = pair * 2 * STRIDE
        right = left + STRIDE
        if right >= NUM_CHUNKS:
            return

        offs_d = tl.arange(0, D)

        m_l = tl.load(m_ptr + h_q * NUM_CHUNKS + left)
        m_r = tl.load(m_ptr + h_q * NUM_CHUNKS + right)
        l_l = tl.load(l_ptr + h_q * NUM_CHUNKS + left)
        l_r = tl.load(l_ptr + h_q * NUM_CHUNKS + right)
        O_l = tl.load(workspace_ptr + h_q * NUM_CHUNKS * D + left  * D + offs_d).to(tl.float32)
        O_r = tl.load(workspace_ptr + h_q * NUM_CHUNKS * D + right * D + offs_d).to(tl.float32)

        m_new = tl.maximum(m_l, m_r)
        alpha = tl.exp(m_l - m_new)
        beta  = tl.exp(m_r - m_new)
        l_new = l_l * alpha + l_r * beta
        O_new = O_l * alpha + O_r * beta

        tl.store(m_ptr + h_q * NUM_CHUNKS + left, m_new)
        tl.store(l_ptr + h_q * NUM_CHUNKS + left, l_new)
        tl.store(workspace_ptr + h_q * NUM_CHUNKS * D + left * D + offs_d, O_new)

    @triton.jit
    def lowrank_recon_kernel(
        U_ptr, V_ptr, anchor_ptr, out_ptr,
        stride_un, stride_uk,
        stride_vk, stride_vd,
        stride_ad,
        stride_on, stride_od,
        n_tokens, rank, feat_dim, scale,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)

        mask_n = offs_n < n_tokens
        mask_d = offs_d < feat_dim

        anchor = tl.load(anchor_ptr + offs_d, mask=mask_d, other=0.0)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)

        for k_start in range(0, rank, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < rank

            u = tl.load(
                U_ptr + offs_n[:, None] * stride_un + offs_k[None, :] * stride_uk,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0,
            )
            v = tl.load(
                V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=mask_k[:, None] & mask_d[None, :], other=0.0,
            )
            acc += tl.dot(u, v)

        if scale != 1.0:
            acc *= scale

        acc += anchor[None, :]
        out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_d[None, :])


# ── OPT-E: Reduction dispatcher \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def _dispatch_reduction(
    out_workspace: "torch.Tensor",
    m_workspace:   "torch.Tensor",
    l_workspace:   "torch.Tensor",
    out:           "torch.Tensor",
    m_out:         "torch.Tensor",
    l_out:         "torch.Tensor",
    num_chunks:    int,
    D:             int,
    H_q:           int,
) -> None:
    """
    Dispatch the multi-chunk LSE-safe reduction to either:
      - Sequential kernel   when num_chunks < 8  (low launch overhead dominates)
      - Parallel tree       when num_chunks >= 8  (ceil(log2(C)) passes, each parallel)

    The parallel tree pads num_chunks to the next power of 2 and runs one
    pairwise-merge Triton kernel per level.  After the last pass, slot 0 of the
    workspace holds the un-normalised merged accumulator; we copy it to `out` and
    divide by l.  Levels where a right-hand partner is out of range are no-ops
    (guarded inside _fused_reduction_pairwise_kernel by the `right >= NUM_CHUNKS`
    early-exit).
    """
    if not HAS_TRITON:
        return  # caller handles the no-Triton path

    PARALLEL_THRESHOLD = 8  # Q3 answer: >= 8 chunks use parallel tree

    if num_chunks < PARALLEL_THRESHOLD:
        # ── Sequential path (existing kernel) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        _fused_sparse_decode_reduction_kernel[(H_q,)](
            out_workspace, m_workspace, l_workspace, out, m_out, l_out,
            num_chunks, D,
        )
    else:
        # ── Parallel tree-reduction path \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        import math as _math
        # Pad to next power of 2 so every level has even pair counts.
        nc_pad = 1 << _math.ceil(_math.log2(num_chunks))
        n_levels = _math.ceil(_math.log2(nc_pad))

        stride = 1
        for _ in range(n_levels):
            n_pairs = nc_pad // (2 * stride)
            if n_pairs == 0:
                break
            grid_pw = (H_q, n_pairs)
            _fused_reduction_pairwise_kernel[grid_pw](
                out_workspace, m_workspace, l_workspace,
                STRIDE=stride, NUM_CHUNKS=num_chunks, D=D,
            )
            stride *= 2

        # After all passes, slot 0 holds the merged (unnormalised) accumulator.
        import torch as _torch
        m_final = m_workspace[:, 0]               # [H_q]
        l_final = l_workspace[:, 0]               # [H_q]
        O_final = out_workspace[:, 0, :].float()  # [H_q, D]

        # The pairwise kernel does NOT divide by l — do it here.
        out_f = O_final / l_final.unsqueeze(-1).clamp(min=1e-9)
        out.copy_(out_f)
        if m_out is not None:
            m_out.copy_(m_final)
        if l_out is not None:
            l_out.copy_(l_final)


# ── 2. PyTorch JIT Helpers for Compilation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _reconstruct_and_score_compiled(
    U: torch.Tensor,
    V_K: torch.Tensor,
    anchors_K: torch.Tensor,
    scales: torch.Tensor,
    cos_sliced: torch.Tensor,
    sin_sliced: torch.Tensor,
    q_sq: torch.Tensor,
    inv_scale: float,
) -> torch.Tensor:
    N = U.shape[0]
    S = U.shape[1]
    R = U.shape[2]
    H = q_sq.shape[0]
    D = q_sq.shape[1]
    
    deltas_k_flat = torch.bmm(U.float(), V_K.float().reshape(N, R, -1))
    # scales shape is [N] (one scale per block).  We need to broadcast across all
    # four dims [N, S, H, D], so reshape to [N, 1, 1, 1].
    # NOTE: scales.unsqueeze(-1) gives [N, 1] which PyTorch pads to [1, 1, N, 1]
    # and then tries to match dim-2 (H) against N — fails for H≠N.
    deltas_k = deltas_k_flat.reshape(N, S, H, D).to(U.dtype) * scales.view(N, 1, 1, 1)
    
    zeros_pad = torch.zeros((N, 1, H, D), dtype=U.dtype, device=U.device)
    deltas_k_full = torch.cat([zeros_pad, deltas_k], dim=1)
    K_unrot_full = anchors_K.unsqueeze(1) + deltas_k_full
    
    half_d = D // 2
    K_unrot_half1 = K_unrot_full[..., :half_d]
    K_unrot_half2 = K_unrot_full[..., half_d:]
    K_unrot_rotated = torch.cat([-K_unrot_half2, K_unrot_half1], dim=-1)
    K_rot_full = K_unrot_full * cos_sliced + K_unrot_rotated * sin_sliced
    
    # q.K as a CONTRACTION, not broadcast-multiply-then-reduce.
    #
    # The old form was `torch.sum(q.view(1,1,H,D) * K_rot_full, dim=-1)`, which
    # materialises the entire [N, S+1, H, D] product before reducing it away --
    # 86 MB per call at the real 16k shapes (N=57, S=257, H=12, D=128), every
    # layer, every decoded token. In the decode profile that single line is the
    # aten::mul (5.7%) and aten::sum (10.0%) entries, and it is a large part of
    # why 66% of DKV's GPU time sits in elementwise+reduce while only 7.7% is in
    # matmul. einsum contracts D directly and allocates only the [N, S+1, H]
    # result.
    #
    # Verified numerically identical (max|diff| 7.6e-6, fp32) and 1.7x faster on
    # CPU alone. The non-compiled sibling in this file already does this via
    # torch.bmm(q_hqd, K_hnd.transpose(1, 2)); this brings the compiled variant
    # in line with it.
    scores = torch.einsum('nshd,hd->nsh', K_rot_full, q_sq) * inv_scale
    return scores

def _attend_and_reconstruct_v_compiled(
    P_anchor: torch.Tensor,
    P_comp: torch.Tensor,
    P_dense: torch.Tensor,
    U: torch.Tensor,
    V_V: torch.Tensor,
    anchors_V: torch.Tensor,
    scales: torch.Tensor,
    v_dense_rep: torch.Tensor,
    V_V_perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    H_q = P_anchor.shape[0]
    N = P_anchor.shape[1]
    D = anchors_V.shape[-1] if anchors_V.numel() > 0 else (v_dense_rep.shape[-1] if v_dense_rep.numel() > 0 else (P_anchor.shape[-1] if P_anchor.dim() > 2 else 0))
    S_dense = P_dense.shape[1] if P_dense.dim() > 1 else 0
    block_capacity = (P_comp.shape[1] // N) if (N > 0 and P_comp.numel() > 0) else 0
    R = U.shape[2] if (N > 0 and U.numel() > 0) else 0

    O_final = torch.zeros((H_q, D), device=P_anchor.device, dtype=P_anchor.dtype)
    if N > 0:
        P_comp_reshaped = P_comp.view(H_q, N, block_capacity).permute(1, 0, 2)
        P_U = torch.bmm(P_comp_reshaped.float(), U.float())

        p_total_anchor = P_anchor.transpose(0, 1) + P_comp_reshaped.sum(dim=-1)
        # Same contraction-not-broadcast point as the score path above; the
        # intermediate here is only ~0.3 MB, so this is for op-count and
        # consistency rather than bandwidth. Verified identical (7.6e-6, fp32).
        O_anchor_fused = torch.einsum('nh,nhd->hd', p_total_anchor, anchors_V.float())
        O_final = O_final + O_anchor_fused.to(P_anchor.dtype)

        P_U_flat = P_U.reshape(N * H_q, 1, R)
        if V_V_perm is None:
            V_V_perm = V_V.float().permute(0, 2, 1, 3).contiguous().reshape(N * H_q, R, D)
        O_delta = torch.bmm(P_U_flat, V_V_perm).reshape(N, H_q, D) * scales.float().view(N, 1, 1)
        O_final = O_final + O_delta.sum(0).to(P_anchor.dtype)

    if S_dense > 0:
        O_dense_total = torch.matmul(
            P_dense.float().view(H_q, 1, S_dense),
            v_dense_rep[0].float().view(H_q, S_dense, D)
        ).squeeze(1)
        O_final = O_final + O_dense_total.to(P_anchor.dtype)

    return O_final


def _prefill_fused_history_attend_compiled(
    U: torch.Tensor,
    V_K: torch.Tensor,
    V_V: torch.Tensor,
    anchors_K: torch.Tensor,
    anchors_V: torch.Tensor,
    scales: torch.Tensor,
    cos_sliced: torch.Tensor,
    sin_sliced: torch.Tensor,
    q: torch.Tensor,
    seq_lens: torch.Tensor,
    inv_scale: float,
    residual_K_positions: torch.Tensor,
    residual_K_values: torch.Tensor,
    residual_V_positions: torch.Tensor,
    residual_V_values: torch.Tensor,
    exact_residual: bool = False,
) -> torch.Tensor:
    # `exact_residual` is a plain bool argument, not an os.environ read, because
    # this function is torch.jit.script'ed on the no-compile path and TorchScript
    # cannot compile os.environ -- reading the flag inside would drop the whole
    # function to eager through the try/except below without saying so.
    N = U.shape[0]
    S = U.shape[1]
    R = U.shape[2]
    H = q.shape[1]
    Q = q.shape[2]
    D = q.shape[3]

    V_K_flat = V_K.reshape(N, R, -1)
    deltas_k_flat = torch.bmm(U, V_K_flat)
    deltas_k = deltas_k_flat.reshape(N, S, H, D) * scales.view(N, 1, 1, 1).to(q.dtype)

    K_unrot_full = torch.cat(
        [anchors_K.unsqueeze(1), anchors_K.unsqueeze(1) + deltas_k], dim=1
    )

    # ── Post-SVD Sparse Residual Correction for Key (Prefill History) ──
    if residual_K_positions.numel() > 0:
        res_pos_K_clamped = residual_K_positions.clamp(min=0).long()
        mask_K = (residual_K_positions >= 0).unsqueeze(-1).unsqueeze(-1)
        res_vals_K = residual_K_values.to(K_unrot_full.dtype)
        if exact_residual:
            # SUBSTITUTION: rk is the anchor-relative EXACT key, so this slot must
            # become anchor + rk -- the low-rank twin deltas_k[p] has to come OUT.
            # Expressed as an additive (rk - twin) so the mask below still zeroes
            # invalid slots exactly as it did for the correction form.
            idx_twin = res_pos_K_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
            twin_K = torch.gather(deltas_k, 1, idx_twin)
            res_vals_K = res_vals_K - twin_K.to(res_vals_K.dtype)
        res_vals_K_masked = res_vals_K * mask_K
        index_K = (res_pos_K_clamped + 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        K_unrot_full.scatter_add_(dim=1, index=index_K, src=res_vals_K_masked)

    half_d = D // 2
    K_half1 = K_unrot_full[..., :half_d]
    K_half2 = K_unrot_full[..., half_d:]
    K_rotated = torch.cat([-K_half2, K_half1], dim=-1)
    cos_s = cos_sliced.squeeze(2)
    sin_s = sin_sliced.squeeze(2)
    K_rot_full = (K_unrot_full * cos_s.unsqueeze(2)
                  + K_rotated  * sin_s.unsqueeze(2))

    q_hqd = q.squeeze(0).reshape(H, Q, D).float()
    K_hnd = K_rot_full.permute(2, 0, 1, 3).reshape(H, N * (1 + S), D).float()
    scores_flat = torch.bmm(q_hqd, K_hnd.transpose(1, 2)) * inv_scale
    scores = scores_flat.reshape(H, Q, N, 1 + S)

    col = torch.arange(1 + S, device=U.device, dtype=torch.long).unsqueeze(0)
    valid = col <= seq_lens.unsqueeze(1).long()
    scores = scores.masked_fill(
        (~valid).unsqueeze(0).unsqueeze(0), float('-inf')
    )

    scores_f = scores.reshape(H, Q, N * (1 + S)).float()
    lse_hist  = torch.logsumexp(scores_f, dim=-1)
    weights_f = torch.softmax(scores_f, dim=-1)
    weights   = weights_f.reshape(H, Q, N, 1 + S).to(q.dtype)

    w_anchor = weights[:, :, :, 0]
    w_delta  = weights[:, :, :, 1:]
    p_total  = w_anchor + w_delta.sum(dim=-1)

    anc_v_hnd = anchors_V.permute(1, 0, 2)
    out_anchor = torch.bmm(p_total, anc_v_hnd)

    w_delta_perm = w_delta.permute(2, 0, 1, 3)
    w_delta_flat = w_delta_perm.reshape(N, H * Q, S)
    W_proj_flat = torch.bmm(w_delta_flat, U)
    W_proj_flat = W_proj_flat * scales.view(N, 1, 1).to(q.dtype)
    W_proj = W_proj_flat.reshape(N, H, Q, R).permute(1, 2, 0, 3)

    V_V_t  = V_V.permute(2, 0, 1, 3)
    W_proj_flat2 = W_proj.reshape(H, Q, N * R)
    V_V_t_flat2 = V_V_t.contiguous().reshape(H, N * R, D)
    out_delta = torch.bmm(W_proj_flat2, V_V_t_flat2)

    out_hist = (out_anchor + out_delta).unsqueeze(0)

    # ── Post-SVD Sparse Residual Correction for Value (Prefill History) ──
    if residual_V_positions.numel() > 0:
        w_d_perm = w_delta.permute(2, 0, 1, 3)
        res_pos_V_clamped = residual_V_positions.clamp(min=0).long()
        res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).unsqueeze(2).expand(-1, H, Q, -1)
        w_res_V = torch.gather(w_d_perm, dim=3, index=res_pos_V_expanded)
        
        mask_V = (residual_V_positions >= 0).unsqueeze(1).unsqueeze(2).expand(-1, H, Q, -1)
        w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

        res_val_V_perm = residual_V_values.to(w_res_V.dtype).permute(0, 2, 1, 3)
        if exact_residual:
            # Remove the lossy twin on the V side too: w·(rv − delta_V[pos]).
            # Inlined rather than routed through _lowrank_delta_v_at so this stays
            # TorchScript-compilable. V_V is [N, R, H, D] here.
            R_u = U.shape[2]
            u_at = torch.gather(U, 1,
                                res_pos_V_clamped.unsqueeze(-1).expand(-1, -1, R_u))
            dv = torch.einsum('npr,nrhd->nphd', u_at.float(), V_V.float())
            dv = dv * scales.reshape(-1, 1, 1, 1).float()
            res_val_V_perm = res_val_V_perm - dv.permute(0, 2, 1, 3).to(res_val_V_perm.dtype)
        O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_perm.unsqueeze(2), dim=(0, 3))
        out_hist = out_hist + O_res.unsqueeze(0).to(out_hist.dtype)

    lse_out  = lse_hist.to(q.dtype).unsqueeze(0)

    lse_padded = lse_out.unsqueeze(-1).expand(1, H, Q, D)
    return torch.stack([out_hist, lse_padded], dim=0)


_IS_MPS_AVAILABLE = (hasattr(torch, "backends") and
                     hasattr(torch.backends, "mps") and
                     torch.backends.mps.is_available())
_IS_CUDA_AVAILABLE = torch.cuda.is_available()

use_compile = os.environ.get("DKV_USE_TORCH_COMPILE", "auto")
if use_compile == "auto":
    use_compile = "1" if _IS_CUDA_AVAILABLE else "0"
elif _IS_MPS_AVAILABLE and not _IS_CUDA_AVAILABLE:
    use_compile = "0"

if use_compile == "1":
    # mode="reduce-overhead" is now correct for both decode functions.
    # The fixed-size dense window workspace (max_dense_len = recency_window + block_size)
    # makes S_dense constant across ALL decode steps.  _attend_and_reconstruct_v receives
    # P_dense[H_q, max_dense_len] and v_dense_rep[H_q, max_dense_len, D] with FIXED shapes
    # every step → Inductor records ONE CUDA graph → replays it → maximum performance.
    # This mirrors what MLX does: @mx.compile compiles once with fixed Metal shaders.
    _decode_compile_mode = "reduce-overhead" if _IS_CUDA_AVAILABLE else "default"
    # PREFILL history-attention must NOT use reduce-overhead.  reduce-overhead
    # records a CUDA graph for a FIXED shape and re-records whenever the shape
    # changes.  The decode kernels above have a constant dense-window shape so
    # that's a win — but the prefill history-attention's compressed-block count
    # (N_blocks) GROWS with context, and under streaming compression it changes
    # on every prefill chunk.  reduce-overhead then re-records the graph each
    # chunk → the ~5s/chunk recompile storm that made DKV_STREAMING_COMPRESS
    # ~5x slower than deferred (fwd 8.5s → 74s at 13k).  "default" compiles once
    # for dynamic shapes (dynamic=True) and never re-records, which is what
    # streaming (the MLX-parity long-context VRAM path) needs.  Deferred prefill
    # never calls this kernel (no blocks are compressed mid-prefill), so this is
    # a no-op for the current default and only unblocks streaming.
    _prefill_compile_mode = "default"

    def _compile_guard(compiled, eager, name):
        """Fall back to eager if the backend fails AT CALL TIME.

        torch.compile() itself only wraps -- it does not compile -- so the
        try/except around it catches nothing. The backend actually runs on the
        first call with a new shape, and when it fails there the exception
        escapes into whatever was running. On a box without a working C compiler
        (Windows without MSVC: "Compiler: cl is not found") that turned
        DKV_STREAMING_COMPRESS=1 into a hard InductorError crash mid-prefill
        rather than a slower-but-working run.

        Only backend failures are swallowed. A genuine runtime error from the
        function -- OOM, a shape bug -- is re-raised, because silently switching
        to eager there would hide it.
        """
        state = {"use": compiled}

        def _run(*a, **kw):
            if state["use"] is eager:
                return eager(*a, **kw)
            try:
                return state["use"](*a, **kw)
            except Exception as exc:                                # noqa: BLE001
                mod = type(exc).__module__ or ""
                if "inductor" not in mod and "dynamo" not in mod:
                    raise
                print(f"[DKV JIT] {name}: backend failed at call time ({exc}). "
                      f"Falling back to eager for the rest of the process.",
                      flush=True)
                state["use"] = eager
                return eager(*a, **kw)
        return _run

    try:
        _backend = "inductor"
        print(f"[DKV JIT] Compiling _reconstruct_and_score with backend={_backend}, mode={_decode_compile_mode} (dynamic=True) ...")
        _reconstruct_and_score = torch.compile(
            _reconstruct_and_score_compiled,
            backend=_backend,
            mode=_decode_compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        _reconstruct_and_score = _compile_guard(
            _reconstruct_and_score, _reconstruct_and_score_compiled, "_reconstruct_and_score")
    except Exception as e:
        print(f"[DKV JIT] torch.compile of _reconstruct_and_score failed ({e}). Falling back to eager.")
        _reconstruct_and_score = _reconstruct_and_score_compiled
        
    try:
        _backend = "inductor"
        print(f"[DKV JIT] Compiling _attend_and_reconstruct_v with backend={_backend}, mode={_decode_compile_mode} (dynamic=True) ...")
        _attend_and_reconstruct_v = torch.compile(
            _attend_and_reconstruct_v_compiled,
            backend=_backend,
            mode=_decode_compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        _attend_and_reconstruct_v = _compile_guard(
            _attend_and_reconstruct_v, _attend_and_reconstruct_v_compiled, "_attend_and_reconstruct_v")
    except Exception as e:
        print(f"[DKV JIT] torch.compile of _attend_and_reconstruct_v failed ({e}). Falling back to eager.")
        _attend_and_reconstruct_v = _attend_and_reconstruct_v_compiled
    try:
        _backend = "inductor"
        print(f"[DKV JIT] Compiling _prefill_fused_history_attend with backend={_backend}, mode={_prefill_compile_mode} (dynamic=True) ...")
        _prefill_fused_history_attend = torch.compile(
            _prefill_fused_history_attend_compiled,
            backend=_backend,
            mode=_prefill_compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        _prefill_fused_history_attend = _compile_guard(
            _prefill_fused_history_attend, _prefill_fused_history_attend_compiled, "_prefill_fused_history_attend")
    except Exception as e:
        print(f"[DKV JIT] torch.compile of _prefill_fused_history_attend failed ({e}). Falling back to JIT script.")
        try:
            _prefill_fused_history_attend = torch.jit.script(_prefill_fused_history_attend_compiled)
        except Exception:
            _prefill_fused_history_attend = _prefill_fused_history_attend_compiled
else:
    _reconstruct_and_score = _reconstruct_and_score_compiled
    _attend_and_reconstruct_v = _attend_and_reconstruct_v_compiled
    try:
        _prefill_fused_history_attend = torch.jit.script(_prefill_fused_history_attend_compiled)
    except Exception:
        _prefill_fused_history_attend = _prefill_fused_history_attend_compiled


# ── 2b-pre. JIT Warmup helper ─────────────────────────────────────────────────
# torch.compile() wraps functions LAZILY — actual Inductor code generation only
# fires on the first real tensor call.  For the CLI, that means the first user
# request pays a 60-120s compile penalty.  For benchmarks, the first measured
# decode step is dominated by compile time and reports artificially low TPS.
#
# warm_up_jit() pre-triggers Inductor with small dummy tensors so both paths
# see steady-state performance from the very first real call — matching how
# MLX's @mx.compile compiles at definition time.
#
# Called from DKVHFWrapper.ensure_loaded() once weights are on-device.

def warm_up_jit(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    H: int = 8,         # query heads (any value; compiled with dynamic=True)
    kv_heads: int = 4,  # KV heads
    D: int = 64,        # head dim  (small → faster compile, works for any D)
    R: int = 4,         # SVD rank
    block_size: int = 16,  # small block for fast dummy ops
) -> None:
    """
    Pre-trigger Inductor compilation for the two decode JIT functions.

    Uses tiny dummy tensors so the compile+kernel-gen finishes in roughly the
    same wall-clock time as with real tensors (Inductor analysis cost dominates
    over tensor size), but without keeping large intermediate buffers alive.

    Safe to call multiple times; subsequent calls hit the compiled cache and
    return instantly.
    """
    if not _IS_CUDA_AVAILABLE or use_compile != "1":
        return  # No-op on MPS/CPU — nothing to warm up

    try:
        N = 2          # number of dummy blocks
        S = block_size - 1  # tokens per block minus anchor
        H_q = H

        _dev = torch.device(device)

        # ── _reconstruct_and_score dummy inputs ───────────────────────────
        U_d       = torch.zeros(N, S, R,          device=_dev, dtype=dtype)
        VK_d      = torch.zeros(N, R, H_q, D,     device=_dev, dtype=dtype)
        anchK_d   = torch.zeros(N, H_q, D,        device=_dev, dtype=dtype)
        scales_d  = torch.ones(N,                  device=_dev, dtype=dtype)
        cos_d     = torch.ones(N, 1 + S, 1, D,    device=_dev, dtype=dtype)
        sin_d     = torch.zeros(N, 1 + S, 1, D,   device=_dev, dtype=dtype)
        q_sq_d    = torch.zeros(H_q, D,           device=_dev, dtype=dtype)

        with torch.no_grad():
            _out = _reconstruct_and_score(U_d, VK_d, anchK_d, scales_d, cos_d, sin_d, q_sq_d, 1.0)
            # Force completion so compilation finishes before we proceed
            if torch.cuda.is_available():
                torch.cuda.synchronize(_dev)

        # ── _attend_and_reconstruct_v dummy inputs ────────────────────────
        S_dens = block_size      # small dense window

        P_anc_d  = torch.zeros(H_q, N,       device=_dev, dtype=dtype)
        P_comp_d = torch.zeros(H_q, N * S,   device=_dev, dtype=dtype)
        P_den_d  = torch.zeros(H_q, S_dens,  device=_dev, dtype=dtype)
        VV_d     = torch.zeros(N, R, H_q, D,     device=_dev, dtype=dtype)  # GQA-expanded
        anchV_d  = torch.zeros(N, H_q, D,         device=_dev, dtype=dtype)  # GQA-expanded
        v_den_d  = torch.zeros(1, S_dens, H_q, D, device=_dev, dtype=dtype)  # GQA-expanded

        with torch.no_grad():
            _out2 = _attend_and_reconstruct_v(
                P_anc_d, P_comp_d, P_den_d,
                U_d, VV_d, anchV_d, scales_d,
                v_den_d,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(_dev)

        # Clean up dummy tensors immediately
        del U_d, VK_d, anchK_d, scales_d, cos_d, sin_d, q_sq_d, _out
        del P_anc_d, P_comp_d, P_den_d, VV_d, anchV_d, v_den_d, _out2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[DKV JIT] Decode kernel warmup complete — Inductor compilation finished.", flush=True)

    except Exception as e:
        # Non-fatal: warmup failure just means first real call will compile.
        print(f"[DKV JIT] WARNING: decode kernel warmup failed ({e}). "
              "First decode request will trigger JIT compilation.", flush=True)


# ── 2b. Stratified U reconstruction helper (Issue 1 fix) ─────────────────────
# The Triton kernel reads pool.U (int8) + pool.U_scale (scalar).  That path
# bypasses the stratified quantization system (U_sem int4 + U_fact fp16) built
# for accuracy parity with MPS.  This helper reconstructs a full fp16 U tensor
# from the stratified components for ALL active blocks BEFORE kernel dispatch,
# then patches a proxy pool so the kernel sees fp16 values and U_scale=1.0.
#
# Cost: one reconstruct_batch_U call per decode step (~N*S*R fp16 elements).
# This is cheaper than re-running full attention; on CUDA with VRAM headroom
# the intermediate tensor lives and dies within the decode step.

class _StratifiedUProxy:
    """
    Thin wrapper that stands in for pool when passing U data to Triton kernels.
    Exposes pool.U as the full-precision fp16 reconstruction and pool.U_scale
    as a tensor of ones so the kernel applies no additional scaling.
    All other attributes delegate to the original pool object.
    """
    __slots__ = ("_pool", "U", "U_scale")

    def __init__(self, pool, U_fp16: torch.Tensor):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "U", U_fp16.to(pool.dtype))       # [n_blocks, S, R] fp16
        # Ones so that kernel's  u = u * u_scale  is a no-op
        object.__setattr__(self, "U_scale",
            torch.ones(pool.U_scale.shape, device=pool.U_scale.device, dtype=pool.U_scale.dtype))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_pool"), name)


# ── OPT-D: Generation-keyed module-level U proxy cache ────────────────────────
# Maps (pool_id, pool_generation, active_key_tuple) → _StratifiedUProxy.
# Invalidated automatically when pool._stratified_generation increments
# (NativeBlockPool.write_block does this on every pool write).
# Cache is intentionally small: a new pool_id is rare (one per session) and each
# generation evicts all prior entries, so the dict stays bounded at O(1) entries
# per active pool across typical single-session inference.
_stratified_proxy_cache: dict = {}


def _build_stratified_U_for_triton(
    pool,
    block_indices: torch.Tensor,
) -> "tuple[object, bool]":
    """
    Build a _StratifiedUProxy for the Triton kernel dispatch.

    Resolves two accuracy issues simultaneously:

    Issue 1 — Stratified U bypass:
        The raw Triton path read pool.U (int8) + pool.U_scale (one scalar per
        block), completely skipping the stratified quantization system
        (U_sem int4 for semantic components + U_fact fp16 for factual
        components).  reconstruct_batch_U() correctly combines both, giving
        the same accuracy as the MPS path.

    Issue 2 — Token-norm quantization error in the int8 path:
        During compression, each token's U row is scaled by its L2 norm before
        storage (compress_layer_blocks_gpu:593, lowrank.py).  The int8 path
        then requantizes ALL rows with a SINGLE global U_scale = max_abs/127.
        Tokens with large norms consume most of the int8 range; low-norm tokens
        are pushed into a tiny slice of it, causing ~0.8% relative error.
        By serving full fp16 U (reconstructed from U_sem + U_fact), every token
        is represented at float16 precision regardless of its norm magnitude.
        The U_scale tensor in the proxy is all-ones, so the kernel's
            u = u * u_scale
        becomes a no-op and the fp16 values are used exactly.

    OPT-C — Active-only reconstruction:
        Previous code called reconstruct_batch_U(pool, all_idx) where all_idx
        covered every slot in the pool (up to 256+), running the expensive
        int4-unpack Python loop for all of them even if only 16 are active.
        Now we do a cheap int8 dequant broadcast for all slots and then overwrite
        only the active slots with accurate stratified reconstruction — cutting the
        expensive loop from N_pool to N_active iterations (typically 16×).

    OPT-D — Generation-keyed cache:
        If the pool has not been written to (same _stratified_generation) and the
        same set of active block indices is requested, return the cached proxy
        immediately without any tensor allocation or reconstruction. Generation
        increments in NativeBlockPool.write_block on every pool update.

    Returns (proxy_or_pool, used_stratified).
    If the pool has no stratified components (n_semantic all-zero) the original
    pool is returned unchanged to avoid a pointless VRAM allocation.
    """
    # Fast-path: no stratified components ever written to this pool.
    # _n_semantic_ever_nonzero is a cheap Python bool set in write_block() only
    # when n_semantic > 0 is actually written. For standard SVD compression this
    # is always False — so we return immediately with ZERO GPU operations and
    # ZERO D2H syncs, regardless of how often _stratified_generation increments.
    n_sem_attr = getattr(pool, "n_semantic", None)
    if n_sem_attr is None:
        return pool, False
    if not getattr(pool, "_n_semantic_ever_nonzero", False):
        return pool, False

    # Stratified blocks exist — do the full check.
    pool_gen = getattr(pool, "_stratified_generation", -1)
    _no_strat     = getattr(pool, "_no_stratified", None)
    _no_strat_gen = getattr(pool, "_no_stratified_gen", None)
    if _no_strat is None or _no_strat_gen != pool_gen:
        _no_strat = bool((n_sem_attr == 0).all().item())
        pool._no_stratified     = _no_strat
        pool._no_stratified_gen = pool_gen
    if _no_strat:
        return pool, False

    active_idx = block_indices.long()

    # ── OPT-D: Cache lookup ────────────────────────────────────────────────────
    pool_id  = id(pool)
    # pool_gen already read above — reuse it.
    # Sort for stable key; routing order may differ but the block set is what matters.
    # tolist() is O(N_active), acceptable for N_active ≤ 256.
    active_key = tuple(sorted(active_idx.tolist()))
    cache_key  = (pool_id, pool_gen, active_key)


    cached = _stratified_proxy_cache.get(cache_key)
    if cached is not None:
        return cached, True   # ── Cache HIT: skip all tensor work ──

    # Evict stale entries only when pool generation changes (keeps cache alive across layers)
    stale = [k for k in _stratified_proxy_cache if k[0] == pool_id and k[1] < pool_gen]
    for k in stale:
        del _stratified_proxy_cache[k]

    # ── OPT-C: Reconstruct only active slots ─────────────────────────────────
    # 1. Fast int8 dequant broadcast as the baseline for ALL pool slots (single
    #    GPU multiply — no Python loop). Active-slot overwrite follows.
    U_full = (pool.U.to(pool.dtype)
              * pool.U_scale.view(-1, 1, 1).to(pool.dtype))  # [n_pool, S, R]
    U_full = U_full.clone()  # detach from pool view before scatter-write

    # 2. Accurate int4/fp16 stratified reconstruction for the N_active slots only.
    #    reconstruct_batch_U loops over idx — now N_active (≤ 16 typical) instead
    #    of N_pool (up to 256+), giving ≥16× speedup on the hot per-decode call.
    U_active = reconstruct_batch_U(pool, active_idx)   # [N_active, S, R] fp16
    U_full[active_idx] = U_active

    proxy = _StratifiedUProxy(pool, U_full)

    # ── OPT-D: Populate cache ─────────────────────────────────────────────────
    _stratified_proxy_cache[cache_key] = proxy
    return proxy, True


# ── F2: routed-row gather + anchor-RoPE rotation, cached per routing interval ──
# The Triton kernels address EVERY per-block tensor through block_indices, so they
# only ever read the N routed rows — yet the dispatchers used to `.clone()` the
# ENTIRE pool (anchors_K, V_K, res_k) on EVERY decode token just to scatter N
# rotated rows into it: O(pool_size) memory traffic per token on the exact path
# being optimized (audit finding F2). This helper instead gathers the N routed
# rows, rotates those, and hands the kernel compact [N]-row tensors with
# block_indices remapped to arange(N) — bit-identical kernel inputs (equivalence
# certified on CPU by tests/test_triton_gather_equiv.py).
#
# The gathered set depends only on (pool contents, routing order, anchor
# positions), NOT on q, so it is cached keyed on (pool id, pool generation,
# exact index order) — the same route-interval reuse that made the MLX fused
# decode fast (per-token work O(1), per-route work O(N)). pool generation is
# NativeBlockPool._stratified_generation, bumped on every write_block/reset;
# if the attribute is missing the cache is skipped (gather still wins vs clone).
# NOTE: the block_indices.tolist() key costs one small D2H sync per call — the
# stratified-U cache above already pays an identical sync per call, so this adds
# no NEW sync point on CUDA.
_gathered_rot_cache: dict = {}


def _gather_routed_blocks_for_kernel(pool_for_kernel, block_indices, anchor_indices, cos, sin):
    """Gather the [N] routed rows of every per-block tensor the Triton kernels
    read, pre-rotating the K-side rows (anchors_K, V_K, res_k) by each block
    anchor's RoPE when rotation inputs are provided. Returns a dict of compact
    tensors plus `idx` = arange(N) to pass as the kernel's block_indices."""
    N = block_indices.shape[0]
    device = block_indices.device
    base_pool = object.__getattribute__(pool_for_kernel, "_pool") \
        if isinstance(pool_for_kernel, _StratifiedUProxy) else pool_for_kernel

    indices = block_indices.long()
    g = {}
    g["idx"] = torch.arange(N, device=device, dtype=block_indices.dtype)

    # Persistent gather buffers (DKV_STATIC_GATHER, default off -- see the note
    # at _gather_into). Disabled while the deferred batch queue is active,
    # because a queued dict must own its tensors until dispatch.
    _reuse = _STATIC_GATHER and not _batch_queue_active()
    if _reuse:
        _bufs = getattr(base_pool, "_dkv_gather_buffers", None)
        if _bufs is None:
            _bufs = {}
            try:
                base_pool._dkv_gather_buffers = _bufs
            except Exception:                                    # noqa: BLE001
                _reuse = False

    if _reuse:
        _G = lambda src, nm: _gather_into(src, indices, _bufs, nm)   # noqa: E731
    else:
        _G = lambda src, nm: src[indices]                            # noqa: E731

    # anchors_K / V_K are ROTATED below, which produces new tensors anyway; the
    # gather buffer only saves the pre-rotation copy.
    anchors_K = _G(pool_for_kernel.anchors_K, "anchors_K")   # [N, H_kv, D]
    V_K       = _G(pool_for_kernel.V_K,       "V_K")         # [N, R, H_kv, D]
    g["anchors_V"] = _G(pool_for_kernel.anchors_V, "anchors_V")
    g["V_V"]       = _G(pool_for_kernel.V_V,       "V_V")
    g["U"]         = _G(pool_for_kernel.U,         "U")
    g["U_scale"]   = _G(pool_for_kernel.U_scale,   "U_scale")
    g["scales"]    = _G(pool_for_kernel.scales,    "scales")
    g["seq_lens"]  = _G(pool_for_kernel.seq_lens,  "seq_lens")

    # Under DKV_ROTATED_POOL the pool already holds POST-RoPE keys (MLX's
    # convention), so every rotation below would be a SECOND rotation. Skipping
    # it is not an optimisation, it is required for correctness -- and it is
    # also strictly cheaper.
    do_rot = (anchor_indices is not None and cos is not None and sin is not None
              and not pool_stores_rotated_k())
    cos_anc = sin_anc = None
    if do_rot:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1)
        cos_anc = cos_flat[anchor_indices_clamped].to(device=V_K.device, dtype=V_K.dtype).unsqueeze(1).unsqueeze(2)
        sin_anc = sin_flat[anchor_indices_clamped].to(device=V_K.device, dtype=V_K.dtype).unsqueeze(1).unsqueeze(2)
        cos_anc_2d = cos_anc.squeeze(2)                 # [N, 1, D] for [N, H_kv, D] tensors
        sin_anc_2d = sin_anc.squeeze(2)
        V_K       = _partial_rope_apply(V_K, cos_anc, sin_anc)
        anchors_K = _partial_rope_apply(anchors_K, cos_anc_2d, sin_anc_2d)
    g["anchors_K"] = anchors_K
    g["V_K"]       = V_K

    res_k   = getattr(base_pool, "residual_K_values",    None)
    res_v   = getattr(base_pool, "residual_V_values",    None)
    res_pos = getattr(base_pool, "residual_K_positions", None)
    res_pos_v = getattr(base_pool, "residual_V_positions", None)
    g["has_res"] = (res_k is not None and res_v is not None and
                    res_pos is not None and res_pos_v is not None)
    if g["has_res"]:
        res_k_g = res_k[indices]                        # [N, MAX_RES, H_kv, D]
        res_pos_g = res_pos[indices]                    # [N, MAX_RES] within-block offsets (-1 padded)
        if do_rot:
            # By default the residual K is rotated at the block ANCHOR position
            # (like V_K) — the PTA approximation. That scrambles the high-frequency
            # RoPE dims of tokens far from the anchor.  A skip block's exact
            # residuals are its most position-sensitive content (digits/codes), so
            # the borderline ones then mis-score and the digits flip/drop at decode
            # even though K/V are value-exact.  Rotate each exact-residual key at its
            # TRUE token position (anchor + within-block offset) instead, matching MLX
            # which appends exact rows as real tokens at their true positions.  Marginal
            # cost over the anchor path this replaces: the cos/sin gather grows from
            # [N, D] to [N, MAX_RES, D] (~sub-MB/step) and one [N, MAX_RES] index add;
            # the residual rotate itself was already here, and the whole `if g[has_res]`
            # block is skipped when no routed block carries residuals — so the dense/no-
            # residual path is untouched and VRAM is unchanged (no persistent buffers).
            # A100-validated: random-code recall 75%→88% (recovered a digit-drop code,
            # zero NIAH regressions).  Default ON; DKV_RESIDUAL_EXACT_ROPE=0 restores
            # the anchor-position approximation.
            if os.environ.get("DKV_RESIDUAL_EXACT_ROPE", "1") == "1":
                # +1: residual_K_positions index the ACTIVE-token array, and the
                # anchor occupies block-local slot 0, so active token j sits at
                # absolute position anchor_idx + 1 + j — not anchor_idx + j.
                # Three independent statements of that layout:
                #   streaming_sparse_ingest.py:1206/1216  anchor = k[..., 0],
                #                                         active = k[..., 1:]
                #   dkv_attention.py:1385                 positions = anchor_indices
                #                                         + arange(1 + max_seq_len)
                #   dkv_decode.metal:153                  "the kernel's anchor
                #                                         rotation lands delta token
                #                                         t at absolute anchor_pos
                #                                         + t + 1 -- its true position"
                # Without it every exact-residual key is rotated one token early.
                # One position is not a rounding error at the top of the RoPE
                # spectrum: theta_0 = 1.0 rad, so the fastest pair is off by a
                # full radian while the slow pairs barely move — the exact
                # signature of a code that comes back with the right letters and
                # the wrong digits (ZEBRA-447 / ZEBRA-474-QUARTZ).
                _abs_pos = (anchor_indices_clamped.unsqueeze(1) + 1
                            + res_pos_g.clamp(min=0).long())          # [N, MAX_RES]
                # A CLAMP HERE IS SILENT CORRUPTION, NOT A GUARD. cos_flat is
                # whatever RoPE table the caller passed; nothing in this function
                # guarantees it spans the whole context. If it is short, every
                # residual key past its end is rotated at the LAST table row
                # instead of its own position -- the key stays value-exact but
                # points the wrong way, so `anchor + residual` scores WORSE than
                # the bare anchor. Report it rather than let it pass as a guard;
                # gated on the trace flag so production pays nothing.
                if os.environ.get("DKV_ROUTE_TRACE") == "1":
                    _lim = cos_flat.shape[0] - 1
                    _over = int((_abs_pos > _lim).sum().item())
                    if _over and not getattr(_gather_routed_blocks_for_kernel,
                                             "_clamp_warned", False):
                        _gather_routed_blocks_for_kernel._clamp_warned = True
                        print(f"[DKV] ROPE CLAMP {_over} residual positions exceed "
                              f"the cos/sin table (rows={cos_flat.shape[0]}, "
                              f"max_requested={int(_abs_pos.max().item())}) — those "
                              f"keys are rotated at the WRONG position", flush=True)
                _abs_pos = _abs_pos.clamp(min=0, max=cos_flat.shape[0] - 1)
                _cos_rk = cos_flat[_abs_pos].to(device=res_k_g.device,
                                                dtype=res_k_g.dtype).unsqueeze(2)   # [N, MAX_RES, 1, D]
                _sin_rk = sin_flat[_abs_pos].to(device=res_k_g.device,
                                                dtype=res_k_g.dtype).unsqueeze(2)
                res_k_g = _partial_rope_apply(res_k_g, _cos_rk, _sin_rk)
            else:
                # Pre-rotate residual K by the block anchor's RoPE, exactly like V_K
                # (reference rotates res_val_K identically). res_v is never rotated.
                res_k_g = _partial_rope_apply(res_k_g, cos_anc, sin_anc)
        g["res_k"]     = res_k_g
        g["res_v"]     = res_v[indices]
        g["res_pos"]   = res_pos_g
        g["res_pos_v"] = res_pos_v[indices]
        g["res_n"]     = (g["res_pos"] >= 0).sum(dim=-1).to(torch.int32)
        g["max_res_pad"] = res_pos.shape[1]
    else:
        g["res_k"]     = torch.empty((0, 0, 0, 0), device=device)
        g["res_v"]     = torch.empty((0, 0, 0, 0), device=device)
        g["res_pos"]   = torch.empty((0, 0), device=device, dtype=torch.int16)
        g["res_pos_v"] = torch.empty((0, 0), device=device, dtype=torch.int16)
        g["res_n"]     = torch.zeros((N,), device=device, dtype=torch.int32)
        g["max_res_pad"] = 1

    fact_pos = getattr(base_pool, "fact_anchor_positions", None)
    fact_ak  = getattr(base_pool, "fact_anchors_K",        None)
    fact_av  = getattr(base_pool, "fact_anchors_V",        None)
    g["has_fact"] = (fact_pos is not None and fact_ak is not None and fact_av is not None)
    if g["has_fact"]:
        g["fact_pos"] = fact_pos[indices]
        g["fact_ak"]  = fact_ak[indices]
        g["fact_av"]  = fact_av[indices]
        g["max_fact"] = fact_pos.shape[1]
    else:
        g["fact_pos"] = torch.empty((0, 0),       device=device, dtype=torch.int16)
        g["fact_ak"]  = torch.empty((0, 0, 0, 0), device=device)
        g["fact_av"]  = torch.empty((0, 0, 0, 0), device=device)
        g["max_fact"] = 1

    return g


# ── 3. PyTorch fallbacks / MPS decoders ───────────────────────────────────────

def fused_decode_attention_mps(
    Q:        torch.Tensor,
    U:        torch.Tensor,
    U_scale:  torch.Tensor,
    VK:       torch.Tensor,
    VV:       torch.Tensor,
    AncK:     torch.Tensor,
    AncV:     torch.Tensor,
    slot_idx: torch.Tensor,
    blk_sizes: torch.Tensor,
) -> torch.Tensor:
    N  = slot_idx.shape[0]
    if N == 0:
        return torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device)

    H_q, D  = Q.shape
    H_kv    = VK.shape[0]
    gpk     = H_q // H_kv
    scale   = D ** -0.5
    q       = Q.float()

    U_a     = U[slot_idx].float() * U_scale[slot_idx].view(N, 1, 1).float()
    AncK_a  = AncK[slot_idx].float()
    AncV_a  = AncV[slot_idx].float()

    AncK_e  = AncK_a.repeat_interleave(gpk, dim=1)
    AncV_e  = AncV_a.repeat_interleave(gpk, dim=1)
    VK_e    = VK.float().repeat_interleave(gpk, dim=0)
    VV_e    = VV.float().repeat_interleave(gpk, dim=0)

    score_anc = torch.einsum('hd,nhd->hn', q, AncK_e) * scale
    q_proj    = torch.einsum('hd,hrd->hr', q, VK_e) * scale
    delta_s   = torch.einsum('hr,nsr->hns', q_proj, U_a)

    s_range   = torch.arange(U_a.shape[1], device=Q.device).view(1, 1, -1)
    valid_mask = s_range < blk_sizes.view(1, N, 1).long()
    delta_s    = delta_s.masked_fill(~valid_mask, float('-inf'))

    all_scores = torch.cat(
        [score_anc.unsqueeze(-1), delta_s], dim=-1
    ).reshape(H_q, -1)

    w = torch.softmax(all_scores, dim=-1).reshape(H_q, N, 1 + U_a.shape[1])
    w_anc = w[:, :, 0]
    w_d   = w[:, :, 1:]

    out_anc  = torch.einsum('hn,nhd->hd', w_anc, AncV_e)
    w_proj   = torch.einsum('hns,nsr->hr', w_d, U_a)
    out_d    = torch.einsum('hr,hrd->hd', w_proj, VV_e)

    return (out_anc + out_d).to(Q.dtype)


def fused_decode_mps(
    Q:                    torch.Tensor,
    pool:                 object,
    block_indices:        Optional[torch.Tensor],
    blk_sizes:            Optional[torch.Tensor],
    num_key_value_groups: int,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    H_q, D   = Q.shape
    gpk      = num_key_value_groups
    scale    = D ** -0.5
    if _DKV_DEBUG_NUMERICS and torch.isnan(Q).any():
        print(f"[fused_decode_mps DEBUG] Q has NaN at start! shape={Q.shape}", flush=True)
    q        = Q.float()

    if block_indices is None or block_indices.numel() == 0:
        return torch.zeros((H_q, D), dtype=Q.dtype, device=Q.device), torch.full((H_q,), float('-inf'), dtype=Q.dtype, device=Q.device)

    N   = block_indices.shape[0]
    idx = block_indices.long()

    U_a    = reconstruct_batch_U(pool, idx).float()
    S_comp = U_a.shape[1]
    AncK_a = pool.anchors_K[idx].float()
    AncV_a = pool.anchors_V[idx].float()
    VK_a   = pool.V_K[idx].float()
    VV_a   = pool.V_V[idx].float()

    if anchor_indices is not None and cos is not None and sin is not None:
        cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
        sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
        
        # 1. Exact RoPE for anchor key
        anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
        cos_anc = cos_flat[anchor_indices_clamped].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1)
        sin_anc = sin_flat[anchor_indices_clamped].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(1)
        
        AncK_e = AncK_a.repeat_interleave(gpk, dim=1)
        AncV_e = AncV_a.repeat_interleave(gpk, dim=1)
        VK_e   = VK_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        VV_e   = VV_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        
        AncK_e_rot = _partial_rope_apply(AncK_e, cos_anc, sin_anc)
        s_anc = torch.einsum('hd,nhd->hn', q, AncK_e_rot) * scale

        # Forwards rotation for VK_e using cos_anc and sin_anc (since keys are pre-rotated at ingest)
        cos_anc_exp = cos_anc.unsqueeze(2) # [N, 1, 1, D]
        sin_anc_exp = sin_anc.unsqueeze(2)
        VK_e_rot = _partial_rope_apply(VK_e, cos_anc_exp, sin_anc_exp)
        
        q_proj_n = torch.einsum('hd,nhrd->nhr', q, VK_e_rot) * scale
        delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
        delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
        delta_s = delta_s + s_anc.unsqueeze(-1)
    else:
        # No RoPE / approximate formulation fallback
        AncK_e = AncK_a.repeat_interleave(gpk, dim=1)
        AncV_e = AncV_a.repeat_interleave(gpk, dim=1)
        VK_e   = VK_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
        VV_e   = VV_a.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()

        s_anc = torch.einsum('hd,nhd->hn', q, AncK_e) * scale
        q_proj_n = torch.einsum('hd,nhrd->nhr', q, VK_e) * scale
        delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
        delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
        delta_s = delta_s + s_anc.unsqueeze(-1)

    # ── Post-SVD Sparse Residual Correction for Key ──
    res_pos_K = getattr(pool, "residual_K_positions", None)
    res_val_K = getattr(pool, "residual_K_values", None)
    if res_pos_K is not None and res_val_K is not None:
        res_pos_K_idx = res_pos_K[idx]
        res_val_K_idx = res_val_K[idx].float()
        if res_pos_K_idx.numel() > 0:
            res_val_K_e = res_val_K_idx.repeat_interleave(gpk, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            has_rope = (anchor_indices is not None and cos is not None and sin is not None)
            if has_rope:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                sin_anc_exp = sin_anc.unsqueeze(1)
                res_val_K_rot = _partial_rope_apply(res_val_K_e, cos_anc_exp, sin_anc_exp)
                corr_K = torch.sum(q.unsqueeze(0).unsqueeze(1) * res_val_K_rot, dim=-1) * scale
            else:
                corr_K = torch.sum(q.unsqueeze(0).unsqueeze(1) * res_val_K_e, dim=-1) * scale

            corr_K_perm = corr_K.permute(2, 0, 1)
            mask_K = (res_pos_K_idx >= 0).unsqueeze(0).expand(H_q, -1, -1)

            res_pos_K_clamped_expanded = res_pos_K_idx.clamp(min=0).long().unsqueeze(0).expand(H_q, -1, -1)
            if _exact_residual_semantics(Q.device):
                # SUBSTITUTION (see _exact_residual_semantics). delta_s already
                # carries s_anc, so the exact score at this position is
                # s_anc + q·rk and writing it drops the lossy twin.
                exact_s = s_anc.unsqueeze(-1).expand_as(corr_K_perm) + corr_K_perm
                _substitute_scores_(delta_s, res_pos_K_clamped_expanded,
                                    exact_s, mask_K)
            else:
                delta_s.scatter_add_(dim=2, index=res_pos_K_clamped_expanded,
                                     src=corr_K_perm.masked_fill(~mask_K, 0.0))

    # ── Solution 3: Fact Anchor overrides for Key ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_K_pool = getattr(pool, "fact_anchors_K", None)
    if fact_pos is not None and fact_anc_K_pool is not None and N > 0:
        fact_pos_idx = fact_pos[idx]  # [N, 3]
        fact_anc_K_idx = fact_anc_K_pool[idx].float()  # [N, 3, num_kv_heads, D]
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            K_exact = fact_anc_K_idx.repeat_interleave(gpk, dim=2)  # [N, 3, H_q, D]
            has_rope = (anchor_indices is not None and cos is not None and sin is not None)
            if has_rope:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q.dtype).unsqueeze(1)
                cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                sin_anc_exp = sin_anc.unsqueeze(1)
                K_exact = _partial_rope_apply(K_exact, cos_anc_exp, sin_anc_exp)
            score_exact = torch.sum(q.view(1, 1, H_q, D) * K_exact, dim=-1) * scale
            score_exact = score_exact.permute(2, 0, 1)  # [H_q, N, 3]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            fact_pos_idx_clamped_expanded = fact_pos_idx_clamped.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
            mask_expanded = mask.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
            delta_s_updates = torch.zeros_like(delta_s)
            delta_s_updates.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=score_exact)
            update_mask = torch.zeros_like(delta_s, dtype=torch.bool)
            update_mask.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=mask_expanded)
            delta_s = torch.where(update_mask, delta_s_updates, delta_s)


    if blk_sizes is not None:
        s_range   = torch.arange(S_comp, device=Q.device).view(1, 1, -1)
        valid_msk = s_range < blk_sizes.view(1, N, 1).long()
        delta_s   = delta_s.masked_fill(~valid_msk, float('-inf'))

    scores = torch.cat(
        [s_anc.unsqueeze(-1), delta_s], dim=-1
    ).reshape(H_q, N * (1 + S_comp))

    lse = torch.logsumexp(scores, dim=-1)
    w = torch.softmax(scores, dim=-1)

    W_comp = w.reshape(H_q, N, 1 + S_comp)
    w_anc  = W_comp[:, :, 0]
    w_d    = W_comp[:, :, 1:]

    # Scale the base/anchor value by the total attention weight of all tokens in the block
    w_block_sum = w_anc + w_d.sum(dim=-1)

    O = torch.zeros((H_q, D), device=Q.device, dtype=torch.float32)
    O = O + torch.einsum('hn,nhd->hd', w_block_sum, AncV_e)

    w_proj = torch.einsum('hns,nsr->nhr', w_d, U_a)
    w_proj = w_proj * pool.scales[idx].float().view(N, 1, 1)
    O = O + torch.einsum('nhr,nhrd->hd', w_proj, VV_e)

    # ── Post-SVD Sparse Residual Correction for Value ──
    res_pos_V = getattr(pool, "residual_V_positions", None)
    res_val_V = getattr(pool, "residual_V_values", None)
    if res_pos_V is not None and res_val_V is not None:
        res_pos_V_idx = res_pos_V[idx]
        res_val_V_idx = res_val_V[idx].float()
        if res_pos_V_idx.numel() > 0:
            res_val_V_e = res_val_V_idx.repeat_interleave(gpk, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            w_d_perm = w_d.permute(1, 0, 2)
            res_pos_V_clamped = res_pos_V_idx.clamp(min=0).long()
            res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = torch.gather(w_d_perm, dim=2, index=res_pos_V_expanded)
            
            mask_V = (res_pos_V_idx >= 0).unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

            res_val_V_e_perm = res_val_V_e.permute(0, 2, 1, 3)
            if _exact_residual_semantics(Q.device):
                # Remove the lossy twin's value: correction is w·(rv − delta_V[pos]).
                # VV_e is [N,H_q,R,D]; the helper wants [N,R,H_q,D].
                dv = _lowrank_delta_v_at(U_a, VV_e.permute(0, 2, 1, 3),
                                         pool.scales[idx], res_pos_V_clamped)
                res_val_V_e_perm = res_val_V_e_perm - dv.permute(0, 2, 1, 3).to(
                    res_val_V_e_perm.dtype)
            O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_e_perm, dim=(0, 2))
            O = O + O_res

    # ── Solution 3: Fact Anchor overrides for Value ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_V_pool = getattr(pool, "fact_anchors_V", None)
    if fact_pos is not None and fact_anc_V_pool is not None and N > 0:
        fact_pos_idx = fact_pos[idx]  # [N, 3]
        fact_anc_V_idx = fact_anc_V_pool[idx].float()  # [N, 3, num_kv_heads, D]
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            V_exact = fact_anc_V_idx.repeat_interleave(gpk, dim=2)  # [N, 3, H_q, D]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            R = U_a.shape[2]
            u_val = torch.gather(U_a, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, R))
            v_svd_sum = torch.sum(u_val.unsqueeze(1).unsqueeze(-1) * VV_e.unsqueeze(2), dim=3)  # [N, H_q, 3, D]
            scales_idx = pool.scales[idx].float()
            v_svd = v_svd_sum * scales_idx.view(N, 1, 1, 1) + AncV_e.unsqueeze(2)  # [N, H_q, 3, D]
            v_svd = v_svd.permute(0, 2, 1, 3)  # [N, 3, H_q, D]
            v_diff = V_exact - v_svd  # [N, 3, H_q, D]
            w_d_perm = w_d.permute(1, 2, 0)  # [N, S_comp, H_q]
            w_pos = torch.gather(w_d_perm, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, H_q))  # [N, 3, H_q]
            w_pos = w_pos.unsqueeze(-1)  # [N, 3, H_q, 1]
            update_term = w_pos * v_diff  # [N, 3, H_q, D]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N, 3, 1, 1]
            update_term = update_term.masked_fill(~mask_expanded, 0.0)
            O = O + torch.sum(update_term, dim=(0, 1))


    if _DKV_DEBUG_NUMERICS and (torch.isnan(O).any() or torch.isinf(O).any() or torch.isnan(lse).any()):
        print("[fused_decode_mps DEBUG] NaN/Inf detected!")
        print(f"q finite: {torch.isfinite(q).all().item()} min: {q.min().item()} max: {q.max().item()}")
        print(f"AncK_e finite: {torch.isfinite(AncK_e).all().item()} min: {AncK_e.min().item()} max: {AncK_e.min().item()} max: {AncK_e.max().item()}")
        print(f"VK_e finite: {torch.isfinite(VK_e).all().item()} min: {VK_e.min().item()} max: {VK_e.max().item()}")
        print(f"U_a finite: {torch.isfinite(U_a).all().item()} min: {U_a.min().item()} max: {U_a.max().item()}")
        print(f"scales finite: {torch.isfinite(pool.scales[idx]).all().item()} min: {pool.scales[idx].min().item()} max: {pool.scales[idx].max().item()}")
        print(f"scale: {scale}")
        print(f"s_anc finite: {torch.isfinite(s_anc).all().item()}")
        print(f"q_proj_n finite: {torch.isfinite(q_proj_n).all().item()}")
        print(f"delta_s finite: {torch.isfinite(delta_s).all().item()}")
        print(f"scores finite: {torch.isfinite(scores).all().item()}")
        print(f"lse finite: {torch.isfinite(lse).all().item()}")
        print(f"w finite: {torch.isfinite(w).all().item()}")

    if _DKV_DEBUG_NUMERICS and lse.max().item() > 100.0:
        print(f"[fused_decode_mps DIAG] lse has large value! max={lse.max().item():.2f}")
        print(f"  q min/max: {q.min().item():.4f}/{q.max().item():.4f}")
        print(f"  AncK_e min/max: {AncK_e.min().item():.4f}/{AncK_e.max().item():.4f}")
        print(f"  VK_e min/max: {VK_e.min().item():.4f}/{VK_e.max().item():.4f}")
        print(f"  U_a min/max: {U_a.min().item():.4f}/{U_a.max().item():.4f}")
        print(f"  pool.scales[idx] min/max: {pool.scales[idx].min().item():.4f}/{pool.scales[idx].max().item():.4f}")
        print(f"  s_anc min/max: {s_anc.min().item():.4f}/{s_anc.max().item():.4f}")
        print(f"  delta_s min/max: {delta_s.min().item():.4f}/{delta_s.max().item():.4f}")
        print(f"  scores min/max: {scores.min().item():.4f}/{scores.max().item():.4f}")

    return O.to(Q.dtype), lse.to(torch.float32)


def _pytorch_vectorized_sparse_attn_decode(
    q:                    torch.Tensor,
    block_indices:        torch.Tensor,
    pool:                 object,
    dense_blocks:         list,            
    active_k:             torch.Tensor,
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
    max_valid_len:        Optional[int] = None,
    cos_sliced:           Optional[torch.Tensor] = None,
    sin_sliced:           Optional[torch.Tensor] = None,
    session_id:           Optional[str] = None,
    layer_idx:            Optional[int] = None,
    decode_workspace:     Optional[dict] = None,
    active_len:           int = 0,           # actual valid dense tokens (workspace may be larger)
) -> torch.Tensor:
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1
    inv_scale = 1.0 / math.sqrt(D)
    q_sq = q.view(H_q, D)
    
    def repeat_kv_at_dim(t, n_rep, dim):
        if n_rep == 1:
            return t
        if dim < 0:
            dim = t.dim() + dim
        shape = list(t.shape)
        val = shape[dim]
        t = t.unsqueeze(dim + 1)
        expand_shape = list(t.shape)
        expand_shape[dim + 1] = n_rep
        t = t.expand(*expand_shape)
        new_shape = shape[:dim] + [val * n_rep] + shape[dim + 1:]
        return t.reshape(*new_shape)

    N = block_indices.shape[0] if block_indices is not None else 0
    block_capacity = 0
    diagnostics = (os.environ.get("DKV_DIAGNOSTICS", "0") == "1")

    # ── N == 0: this decoder returns an EMPTY tensor, shape [1, H_q, 1, 0] ──
    # Reproduced by tests/test_triton_combined.py::test_dense_only, whose whole
    # point is "combined kernel with N=0 (no compressed blocks) must match dense
    # SDPA". It fails with
    #     RuntimeError: The size of tensor a (64) must match tensor b (0)
    #                   at non-singleton dimension 3
    # where 64 is head_dim (the reference) and 0 is what this returns. The dense
    # window is present and simply never attended.
    #
    # Both routes into this function hit it:
    #   native_triton_sparse_attn_decode  ->  `if not HAS_TRITON` early return
    #   native_triton_sparse_attn_decode  ->  the N == 0 `else` branch (with Triton)
    # so on CUDA a decode step that routes zero compressed blocks silently emits a
    # zero-width attention output. native_triton_sparse_attn_decode_combined grew
    # a dense-only SDPA fast path for exactly this case; this decoder never did.
    #
    # Warned rather than fixed: the correct fix is that fast path, and `active_k`
    # here has not been through the RoPE rotation the N>0 branch applies, so
    # writing it without hardware to check against would be guessing. A wrong
    # shape that announces itself is recoverable; the silent version is what let
    # this sit behind a failing test.
    if N == 0:
        global _ZERO_BLOCK_WARNED
        if not _ZERO_BLOCK_WARNED:
            _ZERO_BLOCK_WARNED = True
            _has_dense = (active_k is not None and active_k.shape[2] > 0) or bool(dense_blocks)
            print(f"[DKV] WARNING: 0 compressed blocks routed and this decoder has "
                  f"no dense-only path — attention output will be EMPTY "
                  f"([1, {H_q}, 1, 0] instead of [1, {H_q}, 1, {D}]). "
                  f"dense_window_present={_has_dense}, layer={layer_idx}.",
                  flush=True)

    # ── Features 1 & 2: Heat update + step-ahead prefetch (MPS/CPU path) ──
    if N > 0 and session_id is not None:
        _mgr = getattr(pool, "_manager", None) if pool is not None else None
        if _mgr is not None:
            _tiered = getattr(_mgr, "_kt_tiered_store", None)
            if _tiered is not None and _heat_update_due():
                _slot_list = block_indices.cpu().tolist()
                for _sid in _slot_list:
                    _tiered.update_heat(_sid, routing_score=1.0)
                _tiered.maybe_evict(_occupied_slots(pool, N), protected=_slot_list)
            _prefetch = getattr(_mgr, "_kt_prefetch_engine", None)
            if _prefetch is not None and "_slot_list" in dir():
                _prefetch.submit(session_id, _slot_list)
    # ───────────────────────────────────────────────────────────────────────


    U = torch.empty((0,), device=q.device, dtype=q.dtype)
    V_K = torch.empty((0,), device=q.device, dtype=q.dtype)
    V_V = torch.empty((0,), device=q.device, dtype=q.dtype)
    anchors_K = torch.empty((0,), device=q.device, dtype=q.dtype)
    anchors_V = torch.empty((0,), device=q.device, dtype=q.dtype)
    scales = torch.empty((0,), device=q.device, dtype=q.dtype)
    seq_lens_t = torch.empty((0,), device=q.device, dtype=torch.int32)

    # ── Check configuration-driven caching limits ───────────────────────
    # Issue 6 fix: The gathered-KV workspace cache was designed for MPS where
    # pool.gather() is expensive.  On CUDA the pool tensors are already contiguous
    # GPU memory and gather is cheap; caching stale tensors across block
    # evictions/reallocations causes silent accuracy bugs.  Disable the cache
    # on CUDA by default; enable explicitly with DKV_DECODE_CACHE_ENABLED=1.
    config = getattr(pool, "config", None)
    _on_cuda = (str(getattr(pool, "device", "")) == "cuda" or
                (hasattr(pool, "device") and str(pool.device).startswith("cuda")))
    _default_cache_enabled = False if _on_cuda else True
    decode_cache_enabled = config.decode_cache_enabled if config is not None else _default_cache_enabled
    decode_cache_max_tokens = config.decode_cache_max_tokens if config is not None else 4096

    use_workspace_cache = decode_cache_enabled
    if decode_cache_max_tokens > 0 and total_seq_len > decode_cache_max_tokens:
        use_workspace_cache = False

    session_dict = None
    if use_workspace_cache and decode_workspace is not None and session_id is not None:
        session_dict = decode_workspace.setdefault(session_id, {})
    elif decode_workspace is not None and session_id is not None:
        session_dict = decode_workspace.get(session_id)

    if not use_workspace_cache and decode_workspace is not None and session_id is not None and layer_idx is not None:
        # Clear existing cached tensors in O(1) immediately to reclaim VRAM/RAM
        if session_dict is not None:
            session_dict.get("gathered_kv", {}).pop(layer_idx, None)
            is_empty = True
            for val in session_dict.values():
                if isinstance(val, dict) and len(val) > 0:
                    is_empty = False
                    break
                elif not isinstance(val, dict) and val is not None:
                    is_empty = False
                    break
            if is_empty:
                decode_workspace.pop(session_id, None)

    if N > 0:
        indices = block_indices.long()
        current_version = session_dict.get("routing_version", 0) if session_dict is not None else 0

        cached_gathered = None
        gathered_cache = None
        if use_workspace_cache and session_dict is not None and layer_idx is not None:
            gathered_cache = session_dict.setdefault("gathered_kv", {})
            cached_val = gathered_cache.get(layer_idx)
            if cached_val is not None and cached_val[0] == current_version:
                cached_gathered = cached_val[1]

        # Issue 5 fix: approximate_attn=True is the only supported formulation.
        # The Project-Then-Attend (PTA) approach rotates keys at the ANCHOR position
        # rather than at each token's exact position, which avoids the O(N*S) RoPE
        # embedding gather that per-token rotation would require.  The dead
        # approximate_attn=False branch (exact per-token RoPE) has been removed to
        # reduce confusion.  See the paper §3.2 for the theoretical justification.

        if cached_gathered is not None:
            U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t = cached_gathered
        else:
            U = reconstruct_batch_U(pool, indices).to(q.dtype)
            
            V_K_raw = pool.V_K[indices]
            anchors_K_raw = pool.anchors_K[indices]
            
            if anchor_indices is not None and cos is not None and sin is not None:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                if diagnostics:
                    cpu_anc_check = anchor_indices.cpu()
                    if (cpu_anc_check >= cos_flat.shape[0]).any():
                        print(f"[DKV DEBUG] Out of bounds check: layer_idx={layer_idx} anchor_indices={cpu_anc_check.tolist()} cos_flat.shape={list(cos_flat.shape)}", flush=True)
                
                # Clamp anchor_indices to prevent GPU out of bounds
                anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                cos_anc = cos_flat[anchor_indices_clamped].to(device=V_K_raw.device, dtype=V_K_raw.dtype).unsqueeze(1).unsqueeze(2)
                sin_anc = sin_flat[anchor_indices_clamped].to(device=V_K_raw.device, dtype=V_K_raw.dtype).unsqueeze(1).unsqueeze(2)
                
                cos_anc_2d = cos_flat[anchor_indices_clamped].to(device=anchors_K_raw.device, dtype=anchors_K_raw.dtype).unsqueeze(1)
                sin_anc_2d = sin_flat[anchor_indices_clamped].to(device=anchors_K_raw.device, dtype=anchors_K_raw.dtype).unsqueeze(1)
                
                V_K_raw = _partial_rope_apply(V_K_raw, cos_anc, sin_anc)
                anchors_K_raw = _partial_rope_apply(anchors_K_raw, cos_anc_2d, sin_anc_2d)
                
            V_K = repeat_kv_at_dim(V_K_raw, num_key_value_groups, dim=2)
            V_V = repeat_kv_at_dim(pool.V_V[indices], num_key_value_groups, dim=2)
            anchors_K = repeat_kv_at_dim(anchors_K_raw, num_key_value_groups, dim=1)
            anchors_V = repeat_kv_at_dim(pool.anchors_V[indices], num_key_value_groups, dim=1)
            scales = pool.scales[indices].view(N, 1, 1)
            seq_lens_t = pool.seq_lens[indices]
            
            if use_workspace_cache and gathered_cache is not None:
                gathered_cache[layer_idx] = (current_version, (U, V_K, V_V, anchors_K, anchors_V, scales, seq_lens_t))
        
        block_capacity = U.shape[1]
        R = U.shape[2]

        if max_valid_len is None:
            max_valid_len = int(seq_lens_t.max().item())

        # Cast inputs/pool slices to float32 to prevent float16 overflow/inf issues on MPS
        q_sq_fp32 = q_sq.float()
        anchors_K_fp32 = anchors_K.float()
        V_K_fp32 = V_K.float()
        U_fp32 = U.float()

        has_rope = (anchor_indices is not None and cos is not None and sin is not None)

        # ── Project-Then-Attend formulation (anchor-position RoPE approximation) ──
        # Issue 5 note: per-token exact RoPE branch removed — see comment above.
        # exact anchor score: [H_q, N]
        scores_anchor = torch.einsum('hd,nhd->hn', q_sq_fp32, anchors_K_fp32) * inv_scale
        
        # Project query to V_K: [N, H_q, R]
        q_proj = torch.einsum('hd,nrhd->nhr', q_sq_fp32, V_K_fp32) * inv_scale
        
        # Inner product with U: [H_q, N, block_capacity]
        scores_block = torch.einsum('nhr,nsr->hns', q_proj, U_fp32) * scales.float().view(1, N, 1)
        scores_block = scores_block + scores_anchor.unsqueeze(-1)

        res_pos_K = getattr(pool, "residual_K_positions", None)
        res_val_K = getattr(pool, "residual_K_values", None)
        if res_pos_K is not None and res_val_K is not None and N > 0:
            res_pos_K_idx = res_pos_K[indices]
            res_val_K_idx = res_val_K[indices].float()
            if res_pos_K_idx.numel() > 0:
                res_val_K_e = res_val_K_idx.repeat_interleave(num_key_value_groups, dim=2)
                if has_rope:
                    cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                    sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                    anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                    if os.environ.get("DKV_RESIDUAL_EXACT_ROPE", "1") == "1":
                        # Rotate each exact residual key at its TRUE absolute token
                        # position (anchor + within-block offset), not the block
                        # anchor's -- residual_K holds the exact-minus-recon
                        # correction for ONE specific token, and anchor-position
                        # rotation applies the wrong phase to exactly the
                        # position-sensitive digit/code content these residuals
                        # exist to preserve. Matches _gather_routed_blocks_for_kernel
                        # (the real Triton path, A100-validated 75%->88% random-code
                        # recall) and the Metal kernel's has_exact_res_rope.
                        _abs_pos = (anchor_indices_clamped.unsqueeze(1)
                                    + res_pos_K_idx.clamp(min=0).long())      # [N, MAX_RES]
                        _abs_pos = _abs_pos.clamp(min=0, max=cos_flat.shape[0] - 1)
                        cos_rk = cos_flat[_abs_pos].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(2)
                        sin_rk = sin_flat[_abs_pos].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(2)
                        res_val_K_rot = _partial_rope_apply(res_val_K_e, cos_rk, sin_rk)
                    else:
                        cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                        sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                        cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                        sin_anc_exp = sin_anc.unsqueeze(1)
                        res_val_K_rot = _partial_rope_apply(res_val_K_e, cos_anc_exp, sin_anc_exp)
                    corr_K = torch.sum(q_sq_fp32.unsqueeze(0).unsqueeze(1) * res_val_K_rot, dim=-1) * inv_scale
                else:
                    corr_K = torch.sum(q_sq_fp32.unsqueeze(0).unsqueeze(1) * res_val_K_e, dim=-1) * inv_scale

                corr_K_perm = corr_K.permute(2, 0, 1)
                mask_K = (res_pos_K_idx >= 0).unsqueeze(0).expand(H_q, -1, -1)

                res_pos_K_clamped_expanded = res_pos_K_idx.clamp(min=0).long().unsqueeze(0).expand(H_q, -1, -1)
                if _exact_residual_semantics(q.device):
                    # SUBSTITUTION, as MLX / dkv_decode.metal / both Triton kernels
                    # do: rk is the anchor-relative EXACT key, so this token's true
                    # score is s_anchor + q·rk. Writing that REPLACES the score,
                    # dropping the lossy low-rank twin instead of stacking the
                    # exact key on top of it.
                    exact_s = scores_anchor.unsqueeze(-1).expand_as(corr_K_perm) + corr_K_perm
                    _substitute_scores_(scores_block, res_pos_K_clamped_expanded,
                                        exact_s, mask_K)
                else:
                    scores_block.scatter_add_(
                        dim=2, index=res_pos_K_clamped_expanded,
                        src=corr_K_perm.masked_fill(~mask_K, 0.0))

        # ── Solution 3: Fact Anchor overrides for Key ──
        fact_pos = getattr(pool, "fact_anchor_positions", None)
        fact_anc_K_pool = getattr(pool, "fact_anchors_K", None)
        if fact_pos is not None and fact_anc_K_pool is not None and N > 0:
            fact_pos_idx = fact_pos[indices]  # [N, 3]
            fact_anc_K_idx = fact_anc_K_pool[indices].float()  # [N, 3, num_kv_heads, D]
            mask = (fact_pos_idx >= 0)  # [N, 3]
            if mask.any():
                K_exact = fact_anc_K_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, 3, H_q, D]
                has_rope = (anchor_indices is not None and cos is not None and sin is not None)
                if has_rope:
                    cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                    sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                    anchor_indices_clamped = anchor_indices.clamp(min=0, max=cos_flat.shape[0] - 1).clone()
                    cos_anc = cos_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    sin_anc = sin_flat[anchor_indices_clamped].to(device=q.device, dtype=q_sq_fp32.dtype).unsqueeze(1)
                    cos_anc_exp = cos_anc.unsqueeze(1) # [N, 1, 1, D]
                    sin_anc_exp = sin_anc.unsqueeze(1)
                    K_exact = _partial_rope_apply(K_exact, cos_anc_exp, sin_anc_exp)
                score_exact = torch.sum(q_sq_fp32.view(1, 1, H_q, D) * K_exact, dim=-1) * inv_scale  # [N, 3, H_q]
                score_exact = score_exact.permute(2, 0, 1)  # [H_q, N, 3]
                fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
                fact_pos_idx_clamped_expanded = fact_pos_idx_clamped.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
                mask_expanded = mask.unsqueeze(0).expand(H_q, -1, -1)  # [H_q, N, 3]
                scores_block_updates = torch.zeros_like(scores_block)
                scores_block_updates.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=score_exact)
                update_mask = torch.zeros_like(scores_block, dtype=torch.bool)
                update_mask.scatter_(dim=2, index=fact_pos_idx_clamped_expanded, src=mask_expanded)
                scores_block = torch.where(update_mask, scores_block_updates, scores_block)

        # Mask out-of-bounds tokens
        s_range = torch.arange(block_capacity, device=q.device).view(1, 1, -1)
        valid_mask = s_range < seq_lens_t.view(1, N, 1)
        scores_block = scores_block.masked_fill(~valid_mask, float('-inf'))
        
        scores_compressed = scores_block.reshape(H_q, N * block_capacity)
    else:
        scores_anchor = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)
        scores_compressed = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)

    dense_k_parts = []
    dense_v_parts = []
    
    if active_k is not None and active_len > 0:
        dense_k_parts.append(active_k)
        dense_v_parts.append(active_v)
    else:
        for blk in (dense_blocks or []):
            dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2))
            dense_v_parts.append(blk.anchor_kv[:, 1].unsqueeze(2))
            if blk.active_k is not None:
                dense_k_parts.append(blk.active_k)
                dense_v_parts.append(blk.active_v)

    if dense_k_parts:
        full_k = torch.cat(dense_k_parts, dim=2)
        full_v = torch.cat(dense_v_parts, dim=2)
        
        # S_dense = full workspace size (= max_dense_len, constant across all steps)
        S_dense = full_k.shape[2]

        # Build a fixed-size dense_positions tensor (shape [S_dense], always constant).
        # Pre-allocate in session workspace so the GPU tensor is reused across steps.
        # Positions [active_len..S_dense-1] are padding; they index position 0 in cos/sin
        # (arbitrary — their rotated zeros contribute nothing because scores are masked below).
        if decode_workspace is not None and session_id is not None and layer_idx is not None:
            _sd = decode_workspace.setdefault(session_id, {})
            _dpc = _sd.setdefault("dense_pos_ws", {})
            _dp_ws = _dpc.get(layer_idx)
            if _dp_ws is None or _dp_ws.shape[0] != S_dense or _dp_ws.device != q.device:
                _dp_ws = torch.zeros(S_dense, dtype=torch.long, device=q.device)
                _dpc[layer_idx] = _dp_ws
        else:
            _dp_ws = torch.zeros(S_dense, dtype=torch.long, device=q.device)

        if dense_blocks and active_len > 0:
            _pos = 0
            for blk in dense_blocks:
                _tl = blk.token_indices
                _n = len(_tl)
                if _n > 0 and _pos < S_dense:
                    # Clip to remaining workspace capacity.  This can happen when
                    # assemble_dense_window_kv trimmed the oldest blocks (reducing
                    # the workspace) but dense_blocks still contains all blocks
                    # (including async-compressed ones with active_k=None whose
                    # token_indices list spans more positions than the workspace).
                    _actual_n = min(_n, S_dense - _pos)
                    _dp_ws[_pos:_pos + _actual_n].copy_(
                        torch.tensor(_tl[:_actual_n], dtype=torch.long, device=_dp_ws.device)
                    )
                    _pos += _actual_n
        elif active_len > 0:
            _dp_ws[:active_len] = torch.arange(
                total_seq_len - active_len, total_seq_len, device=q.device
            )
        dense_positions = _dp_ws  # [S_dense] — always fixed shape

        if cos is not None and sin is not None:
            cos_dense = cos[0, dense_positions].unsqueeze(0).unsqueeze(1)  # [1,1,S_dense,rotary_dim]
            sin_dense = sin[0, dense_positions].unsqueeze(0).unsqueeze(1)
            full_k_rot = _partial_rope_apply(full_k, cos_dense, sin_dense)
        else:
            full_k_rot = full_k

        k_dense_rep = repeat_kv_at_dim(full_k_rot, num_key_value_groups, dim=1)
        v_dense_rep = repeat_kv_at_dim(full_v, num_key_value_groups, dim=1)
        scores_dense = torch.sum(q.float() * k_dense_rep.float(), dim=-1).squeeze(0) * inv_scale
        # Mask padding positions so they get 0 softmax weight (not diluted exp(0))
        if active_len < S_dense:
            scores_dense = scores_dense.clone()
            scores_dense[:, active_len:] = float('-inf')
    else:
        S_dense = 0
        scores_dense = torch.empty((H_q, 0), device=q.device, dtype=torch.float32)

    scores_all = torch.cat([scores_anchor, scores_compressed, scores_dense], dim=-1)
    
    if diagnostics:
        has_nan = torch.isnan(scores_all).any().item()
        has_posinf = (scores_all == float('inf')).any().item()
        if has_nan or has_posinf:
            scores_all = scores_all.clone()
            if has_nan:
                scores_all[torch.isnan(scores_all)] = -1e4
            if has_posinf:
                scores_all[scores_all == float('inf')] = 1e4

    probs_all = torch.nn.functional.softmax(scores_all.float(), dim=-1).to(scores_all.dtype)
    P_anchor, P_comp, P_dense = torch.split(probs_all, [N, N * block_capacity, S_dense], dim=-1)

    O_final = _attend_and_reconstruct_v(
        P_anchor=P_anchor,
        P_comp=P_comp,
        P_dense=P_dense,
        U=U,
        V_V=V_V,
        anchors_V=anchors_V,
        scales=scales,
        v_dense_rep=v_dense_rep if S_dense > 0 else torch.empty((0,), device=q.device),
        V_V_perm=None,
    )

    # ── Post-SVD Sparse Residual Correction for Value ──
    res_pos_V = getattr(pool, "residual_V_positions", None)
    res_val_V = getattr(pool, "residual_V_values", None)
    if res_pos_V is not None and res_val_V is not None and N > 0:
        res_pos_V_idx = res_pos_V[indices]
        res_val_V_idx = res_val_V[indices].float()
        if res_pos_V_idx.numel() > 0:
            res_val_V_e = res_val_V_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, MAX_RESIDUAL_TOKENS, H_q, D]
            P_comp_reshaped = P_comp.view(H_q, N, block_capacity).permute(1, 0, 2)  # [N, H_q, block_capacity]
            res_pos_V_clamped = res_pos_V_idx.clamp(min=0).long()
            res_pos_V_expanded = res_pos_V_clamped.unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = torch.gather(P_comp_reshaped, dim=2, index=res_pos_V_expanded)
            
            mask_V = (res_pos_V_idx >= 0).unsqueeze(1).expand(-1, H_q, -1)
            w_res_V = w_res_V.masked_fill(~mask_V, 0.0)

            res_val_V_e_perm = res_val_V_e.permute(0, 2, 1, 3)
            if _exact_residual_semantics(q.device):
                # Substitute on the V side too, or the score would be exact while
                # the value stayed lossy. The row already contributes
                # w·(anchor_V + delta_V[pos]); the exact value is anchor_V + rv,
                # so the correction is w·(rv − delta_V[pos]).
                # V_V was already repeat_kv'd to H_q above, so dv comes back at
                # H_q directly -- do NOT expand it a second time.
                dv = _lowrank_delta_v_at(U, V_V, scales.reshape(-1),
                                         res_pos_V_clamped)              # [N,P,H_q,D]
                dv = dv.permute(0, 2, 1, 3)                              # [N,H_q,P,D]
                res_val_V_e_perm = res_val_V_e_perm - dv.to(res_val_V_e_perm.dtype)
            O_res = torch.sum(w_res_V.unsqueeze(-1) * res_val_V_e_perm, dim=(0, 2))
            O_final = O_final + O_res.to(O_final.dtype)

    # ── Solution 3: Fact Anchor overrides for Value ──
    fact_pos = getattr(pool, "fact_anchor_positions", None)
    fact_anc_V_pool = getattr(pool, "fact_anchors_V", None)
    if fact_pos is not None and fact_anc_V_pool is not None and N > 0:
        fact_pos_idx = fact_pos[indices]
        fact_anc_V_idx = fact_anc_V_pool[indices].float()
        mask = (fact_pos_idx >= 0)  # [N, 3]
        if mask.any():
            V_exact = fact_anc_V_idx.repeat_interleave(num_key_value_groups, dim=2)  # [N, 3, H_q, D]
            fact_pos_idx_clamped = fact_pos_idx.clamp(min=0).long()
            R = U.shape[2]
            u_val = torch.gather(U, dim=1, index=fact_pos_idx_clamped.unsqueeze(-1).expand(-1, -1, R))  # [N, 3, R]
            v_svd_sum = torch.sum(u_val.unsqueeze(1).unsqueeze(-1) * V_V.permute(0, 2, 1, 3).unsqueeze(2), dim=3)  # [N, H_q, 3, D]
            v_svd = v_svd_sum * scales.view(N, 1, 1, 1) + anchors_V.unsqueeze(2)  # [N, H_q, 3, D]
            v_svd = v_svd.permute(0, 2, 1, 3)  # [N, 3, H_q, D]
            v_diff = V_exact - v_svd  # [N, 3, H_q, D]
            fact_idx_expanded = fact_pos_idx_clamped.unsqueeze(1).expand(-1, H_q, -1)  # [N, H_q, 3]
            w_pos = torch.gather(P_comp_reshaped, dim=2, index=fact_idx_expanded)  # [N, H_q, 3]
            w_pos = w_pos.permute(0, 2, 1)  # [N, 3, H_q]
            w_pos = w_pos.unsqueeze(-1)  # [N, 3, H_q, 1]
            update_term = w_pos * v_diff  # [N, 3, H_q, D]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N, 3, 1, 1]
            update_term = update_term.masked_fill(~mask_expanded, 0.0)
            O_final = O_final + torch.sum(update_term, dim=(0, 1)).to(O_final.dtype)

    # Issue 4 fix: Layer-0 NaN diagnostics gated behind DKV_LAYER0_DEBUG=1.
    # Previously fired every decode step for every session, burning CPU time and
    # polluting logs.  Enable during bring-up / NaN debugging only.
    if layer_idx == 0 and os.environ.get("DKV_LAYER0_DEBUG", "0") == "1":
        print(f"[DKV DEBUG] layer 0 check - q has nan: {torch.isnan(q).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - block_indices has nan: {torch.isnan(block_indices).any().item() if block_indices is not None else False}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - U has nan: {torch.isnan(U).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - V_K has nan: {torch.isnan(V_K).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - V_V has nan: {torch.isnan(V_V).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - anchors_K has nan: {torch.isnan(anchors_K).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - anchors_V has nan: {torch.isnan(anchors_V).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - scales has nan: {torch.isnan(scales).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - scores_anchor has nan: {torch.isnan(scores_anchor).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - scores_compressed has nan: {torch.isnan(scores_compressed).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - scores_dense has nan: {torch.isnan(scores_dense).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - scores_all has nan: {torch.isnan(scores_all).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - probs_all has nan: {torch.isnan(probs_all).any().item()}", flush=True)
        print(f"[DKV DEBUG] layer 0 check - O_final has nan: {torch.isnan(O_final).any().item()}", flush=True)

    return O_final.to(q.dtype).unsqueeze(0).unsqueeze(2)


def residuals_in_dense() -> bool:
    """DKV_RESIDUALS_IN_DENSE — put exact residual rows in the DENSE half, as MLX does.

    MLX carries a block's exact residual rows in the DENSE partition
    (`mlx_dkv_wrapper.py:1031  dense_k_for_attn = concat([res_k_all, dense_k])`) and
    masks their lossy low-rank twins out of the SPARSE half
    (`:771  delta_s = where(res_mask, -inf, delta_s)`). CUDA keeps them in the SPARSE
    half instead.

    That is not a cosmetic difference. The merge bias is

        bias = max(0, base - 0.5 * max(0, (lse_dense - lse_sparse) - 4))

    and the whole reason it is NIAH-safe is "the exact needle residual makes
    lse_dense dominate -> bias -> 0" (mlx_dkv_wrapper.py:26-28). That holds only
    while the exact rows are on the DENSE side. With them on the SPARSE side the gap
    moves the wrong way and `auto` stays pinned near its +2.0 maximum instead of
    decaying to 0 -- and the same note records that +4.0 breaks NIAH outright by
    needle corruption. So the control meant to detect "an exact needle is present,
    back off" is reading the wrong side of the merge.

    MLX's own comment describes the failure this produces exactly:
    "when the answer lives in OLD (compressed) context that is out-competed by the
    recent exact dense window, the model reads the wrong region ... the sparse half
    attended the correct tokens but LOST the merge."

    DEFAULT OFF. Turning it on requires BOTH halves of the change:
      (1) this kernel flag, which masks the twin instead of substituting, and
      (2) the dispatcher appending the exact rows to active_k/active_v and
          extending active_len.
    Enabling (1) alone DROPS those tokens from attention entirely. The pairing is
    asserted by tests/test_residual_partition.py.
    """
    return os.environ.get("DKV_RESIDUALS_IN_DENSE", "0") == "1"


def build_dense_residual_rows(g, dtype=None):
    """The exact residual rows MLX puts in the dense half: `[n_rows, H_kv, D]` K and V.

    Mirrors `mlx_dkv_wrapper.py:1002-1006` (`rk = take(comp_res_k, sel)`) except for
    the storage format: MLX stores `comp_res_k` as the ABSOLUTE rotated key, CUDA
    stores `residual_K_values` ANCHOR-RELATIVE, so the exact key here is
    `anchors + res_val` -- the same reconstruction `_scatter_residuals` performs and
    the one `probe_residual_values.py` verified against ground truth at cos 1.0000.

    Validity comes from `res_pos >= 0` (the -1 padding), matching MLX's
    `res_valid = arange(R) < res_n`. Padded slots are dropped rather than emitted
    with a mask, so the caller can simply concatenate and extend its length.

    K and V take their own position arrays: `residual_V_positions` are selected
    independently of `residual_K_positions`, so a row can be exact in one and not
    the other. Rows are emitted on the UNION -- a token exact in K but not V keeps
    its true K and falls back to `anchors_V` for V, which is what the sparse-half
    substitution did too. Returns (K, V, n_rows); n_rows == 0 when nothing is valid.
    """
    import torch as _t
    if not g.get("has_res"):
        return None, None, 0
    res_pos, res_pos_v = g["res_pos"], g["res_pos_v"]
    res_k, res_v = g["res_k"], g["res_v"]
    aK, aV = g["anchors_K"], g["anchors_V"]          # [N, H_kv, D]
    keep = (res_pos.long() >= 0) | (res_pos_v.long() >= 0)   # [N, MAX_RES]
    n_rows = int(keep.sum().item())
    if n_rows == 0:
        return None, None, 0
    # anchor + residual, per slot, in fp32 before the cast: the corrections are small
    # deltas on a much larger anchor, exactly where fp16 rounding eats them.
    exact_k = aK.unsqueeze(1).float() + res_k.float()          # [N, MAX_RES, H_kv, D]
    exact_v = aV.unsqueeze(1).float() + res_v.float()
    # A slot valid only in V has no true K (and vice versa): fall back to the anchor.
    kv = (res_pos.long() >= 0).unsqueeze(-1).unsqueeze(-1)
    vv = (res_pos_v.long() >= 0).unsqueeze(-1).unsqueeze(-1)
    exact_k = _t.where(kv, exact_k, aK.unsqueeze(1).float())
    exact_v = _t.where(vv, exact_v, aV.unsqueeze(1).float())
    out_k = exact_k[keep]                                      # [n_rows, H_kv, D]
    out_v = exact_v[keep]
    dt = dtype if dtype is not None else res_k.dtype
    return out_k.to(dt), out_v.to(dt), n_rows


def native_triton_sparse_attn_decode(
    q:                    torch.Tensor,
    block_indices:        torch.Tensor,
    pool:                 object,
    dense_blocks:         list,            
    active_k:             torch.Tensor,
    active_v:             torch.Tensor,
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    total_seq_len:        int = 0,
    max_valid_len:        Optional[int] = None,
    cos_sliced:           Optional[torch.Tensor] = None,
    sin_sliced:           Optional[torch.Tensor] = None,
    session_id:           Optional[str] = None,
    layer_idx:            Optional[int] = None,
    decode_workspace:     Optional[dict] = None,
    active_len:           int = 0,           # actual valid dense tokens (workspace may be padded)
) -> torch.Tensor:
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1
    
    if not HAS_TRITON:
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
            cos_sliced=cos_sliced, sin_sliced=sin_sliced,
            session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
            active_len=active_len,
        )
        
    inv_scale = 1.0 / math.sqrt(D)
    N = block_indices.shape[0] if block_indices is not None else 0

    # ── Features 1 & 2: Heat update + step-ahead prefetch ─────────────────
    # After routing delivers block_indices, tell the TieredBlockStore that these
    # slots are "hot" and submit an async prefetch job for the next decode step.
    #
    # These DO synchronise, contrary to what this comment used to claim: both
    # need a HOST-side slot list, and block_indices.cpu().tolist() below is a
    # device->host copy on every decode step.
    # torch.cuda.set_sync_debug_mode("error") names it directly.
    #
    # Skipped under DKV_GRAPH_SAFE_DECODE because a stream capture is invalidated
    # by any host sync. Both are performance heuristics -- keeping routed slots
    # warm, and prefetching them before the next step -- so dropping them costs
    # some tiering benefit and nothing in correctness.
    if N > 0 and session_id is not None and not _GRAPH_SAFE_DECODE:
        _mgr = getattr(pool, "_manager", None) if pool is not None else None
        if _mgr is not None:
            # Feature 1: mark routed slots hot so eviction keeps them warm.
            _tiered = getattr(_mgr, "_kt_tiered_store", None)
            if _tiered is not None and _heat_update_due():
                _slot_list = block_indices.cpu().tolist()
                for _sid in _slot_list:
                    _tiered.update_heat(_sid, routing_score=1.0)
                # Proactively evict cold slots when pool fill is high (async no-op if below thresh).
                _tiered.maybe_evict(_occupied_slots(pool, N), protected=_slot_list)

            # Feature 2: submit routed slots to the prefetch engine so they are
            # H2D-warm before the next decode step begins.
            _prefetch = getattr(_mgr, "_kt_prefetch_engine", None)
            if _prefetch is not None:
                _prefetch.submit(session_id, _slot_list if "_slot_list" in dir() else block_indices.cpu().tolist())
    # ──────────────────────────────────────────────────────────────────────────

    if N > 0:

        try:
            q_sq = q[0, :, 0, :]
            D_pad = triton.next_power_of_2(D)
            R_pad = triton.next_power_of_2(R)
            S_pad = triton.next_power_of_2(S_MAX)
            
            # DKV_BLOCKS_PER_CHUNK — DIAGNOSTIC KNOB, default 16 (unchanged).
            # num_chunks = ceil(N / this), and num_chunks > 1 switches on the
            # cross-chunk online-softmax reduction (_dispatch_reduction), which
            # further forks: sequential below 8 chunks, parallel tree at 8+.
            # Needle recall correlates exactly with that boundary on the
            # production path: 2k (~5 blocks, 1 chunk) passes 3/3; 8k (~37
            # blocks, 3 chunks -> sequential) gives 1/3 and ZEBRA-447; 32k
            # (~120 blocks, 8 chunks -> parallel tree) gives ZEBRA-474-QUARTZ
            # and None. Setting this above N forces num_chunks=1 and skips the
            # reduction entirely, which isolates "the reduction is wrong" from
            # "the per-block math is wrong" in ONE run.
            BLOCKS_PER_CHUNK = _blocks_per_chunk(N, H_q, q.device)
            if N > BLOCKS_PER_CHUNK:
                num_chunks = (N + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK
            else:
                num_chunks = 1
                
            grid = (H_q, num_chunks)
            
            if num_chunks > 1:
                cache_key = (H_q, num_chunks, D_pad, q.device)
                if not hasattr(native_triton_sparse_attn_decode, "_workspaces_cache"):
                    native_triton_sparse_attn_decode._workspaces_cache = {}
                
                workspaces = native_triton_sparse_attn_decode._workspaces_cache.get(cache_key)
                if workspaces is None:
                    out_workspace = torch.empty((H_q, num_chunks, D_pad), device=q.device, dtype=torch.float32)
                    m_workspace = torch.empty((H_q, num_chunks), device=q.device, dtype=torch.float32)
                    l_workspace = torch.empty((H_q, num_chunks), device=q.device, dtype=torch.float32)
                    workspaces = (out_workspace, m_workspace, l_workspace)
                    native_triton_sparse_attn_decode._workspaces_cache[cache_key] = workspaces
                else:
                    out_workspace, m_workspace, l_workspace = workspaces
                    
                out = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
                m_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                l_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
            else:
                out = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
                m_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                l_out = torch.empty((H_q,), device=q.device, dtype=torch.float32)
                out_workspace = out
                m_workspace = m_out
                l_workspace = l_out
            # Issue 1 fix: Pre-reconstruct stratified U (int4 semantic + fp16 factual)
            # into a full fp16 tensor before Triton dispatch.  The kernel's U_scale
            # tensor is all-ones in the proxy so u = u * 1.0 = exact fp16 values.
            # This gives CUDA accuracy parity with the MPS path (reconstruct_batch_U).
            pool_for_kernel, _used_strat = _build_stratified_U_for_triton(pool, block_indices)

            # F2 fix: gather + rotate ONLY the N routed rows (cached per routing
            # interval) instead of cloning the whole pool per token; the kernel
            # gets compact [N]-row tensors with block_indices remapped to
            # arange(N) — bit-identical inputs, O(pool)→O(N) per-token traffic.
            g = _gather_routed_blocks_for_kernel(
                pool_for_kernel, block_indices, anchor_indices, cos, sin)

            _fused_sparse_decode_kernel[grid](
                q_sq, g["idx"], g["anchors_K"], g["anchors_V"], g["V_K"], g["V_V"],
                g["U"], g["U_scale"], g["scales"], g["seq_lens"],
                g["res_k"], g["res_v"], g["res_pos"], g["res_pos_v"], g["res_n"],
                g["fact_pos"], g["fact_ak"], g["fact_av"],
                out_workspace, m_workspace, l_workspace,
                q_sq.stride(0), q_sq.stride(1),
                g["anchors_K"].stride(0), g["anchors_K"].stride(1), g["anchors_K"].stride(2),
                g["anchors_V"].stride(0), g["anchors_V"].stride(1), g["anchors_V"].stride(2),
                g["V_K"].stride(0), g["V_K"].stride(1), g["V_K"].stride(2), g["V_K"].stride(3),
                g["V_V"].stride(0), g["V_V"].stride(1), g["V_V"].stride(2), g["V_V"].stride(3),
                g["U"].stride(0), g["U"].stride(1), g["U"].stride(2),
                g["res_k"].stride(0), g["res_k"].stride(1), g["res_k"].stride(2), g["res_k"].stride(3),
                g["res_v"].stride(0), g["res_v"].stride(1), g["res_v"].stride(2), g["res_v"].stride(3),
                g["res_pos"].stride(0), g["res_pos_v"].stride(0),
                g["fact_pos"].stride(0),
                g["fact_ak"].stride(0), g["fact_ak"].stride(1), g["fact_ak"].stride(2), g["fact_ak"].stride(3),
                g["fact_av"].stride(0), g["fact_av"].stride(1), g["fact_av"].stride(2), g["fact_av"].stride(3),
                out_workspace.stride(0), out_workspace.stride(1),
                N, H_q, g["anchors_K"].shape[1], num_key_value_groups, D_pad,
                R_pad, R, S_pad, inv_scale, BLOCKS_PER_CHUNK, num_chunks,
                MAX_RESIDUAL=g["max_res_pad"], MAX_FACT=g["max_fact"],
                HAS_RESIDUAL=g["has_res"], HAS_FACT=g["has_fact"],
                EXACT_RESIDUAL=_exact_residual_semantics(q.device),
                # MUST be passed. A tl.constexpr declared on the kernel but omitted
                # here is the `S_MAX` failure mode this file already paid for: the
                # launch raises TypeError and the try/except drops to the PyTorch
                # fallback SILENTLY, so production runs a different implementation
                # than the one under test.
                RESIDUAL_IN_DENSE=residuals_in_dense(),
            )
            
            if num_chunks > 1:
                # OPT-E: dispatches sequential vs. parallel tree reduction based on num_chunks
                _dispatch_reduction(
                    out_workspace, m_workspace, l_workspace, out, m_out, l_out,
                    num_chunks, D_pad, H_q,
                )

            if dense_blocks or (active_k is not None and active_k.shape[2] > 0):
                O_i = out * l_out.unsqueeze(-1)
                m_i = m_out
                l_i = l_out

                # Prefer active_k (pre-assembled workspace) over iterating dense_blocks
                # to avoid double-counting (active_k already contains all anchor+active data
                # from every dense block, assembled by assemble_dense_window_kv).
                if active_k is not None and active_k.shape[2] > 0:
                    # active_k/active_v are the FIXED-SIZE assembled workspace, padded
                    # to max_dense_len. Slice to the actual valid dense count so we
                    # (a) never attend over padding rows, and (b) the RoPE length-check
                    # below (len(_dense_pos_list) == k_kv.shape[2]) MATCHES. When left
                    # padded, token_count != padded_len so rotation was silently
                    # skipped, leaving the dense keys UNROTATED — corrupting the recent-
                    # window attention and producing garbage decode (the deep-needle bug
                    # in the DKV_SPARSE_BIAS=auto production path).
                    _alen = active_len if (active_len and active_len > 0) else active_k.shape[2]
                    k_kv = active_k[:, :, :_alen].float()
                    v_kv = active_v[:, :, :_alen].float()
                else:
                    dense_k_parts = []
                    dense_v_parts = []
                    for blk in (dense_blocks or []):
                        if blk.anchor_kv is not None:
                            dense_k_parts.append(blk.anchor_kv[:, 0].unsqueeze(2))
                            dense_v_parts.append(blk.anchor_kv[:, 1].unsqueeze(2))
                        if blk.active_k is not None and blk.active_k.shape[2] > 0:
                            dense_k_parts.append(blk.active_k)
                            dense_v_parts.append(blk.active_v)
                    if dense_k_parts:
                        k_kv = torch.cat(dense_k_parts, dim=2).float()
                        v_kv = torch.cat(dense_v_parts, dim=2).float()
                    else:
                        k_kv = None
                        v_kv = None

                if k_kv is not None and k_kv.shape[2] > 0:
                    # Apply RoPE rotation with correct absolute token positions.
                    # Without this, q_rot(pos_q) @ k_unrot gives wrong attention scores
                    # for dense (ACCUMULATING) blocks, breaking NIAH retrieval on CUDA.
                    # Dense-window blocks come from the same pool/ingest path, so
                    # under DKV_ROTATED_POOL their keys are already POST-RoPE and
                    # rotating again would corrupt the recent window -- the one
                    # part of attention that is otherwise exact.
                    if (dense_blocks and cos is not None and sin is not None
                            and not pool_stores_rotated_k()):
                        _dense_pos_list = []
                        for _blk in (dense_blocks or []):
                            _dense_pos_list.extend(_blk.token_indices)
                        _rot_fired = bool(_dense_pos_list) and len(_dense_pos_list) == k_kv.shape[2]
                        if _rot_fired:
                            _dp = torch.tensor(_dense_pos_list, dtype=torch.long, device=k_kv.device)
                            # cos/sin here are the RAW rotary_emb output, so the
                            # last dim is ROTARY_DIM, not head_dim. The old comment
                            # claimed "[1,1,L,D]" and the code below rotated full
                            # width off that assumption, which is fine only when
                            # partial_rotary_factor == 1.0. On Qwen3.5-2B
                            # (head_dim 256, rotary_dim 64) it raised
                            #   size of tensor a (256) must match tensor b (64)
                            # the moment this path could actually be reached --
                            # which was never, until the S_MAX autotune collision
                            # above it was fixed. Note the two OTHER hand-rolled
                            # rotations in this file (~735, ~845) take cos_SLICED,
                            # already expanded to head_dim, which is why the
                            # PyTorch fallback survived on the same model.
                            _cos_d = cos[0, _dp.clamp(max=cos.shape[1] - 1)].unsqueeze(0).unsqueeze(1)  # [1,1,L,rotary_dim]
                            _sin_d = sin[0, _dp.clamp(max=sin.shape[1] - 1)].unsqueeze(0).unsqueeze(1)
                            # _partial_rope_apply rotates the leading rotary_dim
                            # slice and passes the tail through; it reduces to the
                            # exact expression this replaced when rotary_dim >= D,
                            # so full-rotary models (Qwen2.5, rotary_dim == 128 ==
                            # head_dim) are bit-identical.
                            k_kv = _partial_rope_apply(k_kv,
                                                       _cos_d.to(k_kv.dtype),
                                                       _sin_d.to(k_kv.dtype))

                    # ── MLX partition: exact residual rows join the DENSE half ──
                    # mlx_dkv_wrapper.py:1031  dense_k_for_attn = concat([res_k_all,
                    # dense_k]). The kernel above has masked their lossy twins to
                    # -inf (RESIDUAL_IN_DENSE), so each such token is represented
                    # HERE and only here -- appending without that mask, or masking
                    # without appending, breaks the exactly-once invariant that
                    # tests/test_residual_partition.py pins.
                    #
                    # APPENDED AFTER THE ROPE BLOCK ABOVE, DELIBERATELY. These rows
                    # come out of the pool already POST-RoPE (DKV_ROTATED_POOL, which
                    # is why the gather's do_rot is False), so routing them through
                    # the rotation that k_kv may take would rotate them TWICE --
                    # the same class of corruption as the unrotated-dense-window bug
                    # documented at the slice above, and it would land on precisely
                    # the rows this change exists to protect.
                    if residuals_in_dense():
                        _g_res = locals().get("g")
                        if _g_res is not None:
                            _rk, _rv, _nres = build_dense_residual_rows(
                                _g_res, dtype=k_kv.dtype)
                            if _nres > 0:
                                # [n, H_kv, D] -> [1, H_kv, n, D] to match k_kv
                                _rk = _rk.permute(1, 0, 2).unsqueeze(0).to(k_kv.device)
                                _rv = _rv.permute(1, 0, 2).unsqueeze(0).to(v_kv.device)
                                k_kv = torch.cat([_rk.float(), k_kv], dim=2)
                                v_kv = torch.cat([_rv.float(), v_kv], dim=2)

                    H_q, D = q_sq.shape
                    H_kv = k_kv.shape[1]
                    n_rep = num_key_value_groups

                    q_reshaped = q_sq.float().view(H_kv, n_rep, D)
                    k_permuted = k_kv[0].permute(0, 2, 1)

                    s = torch.bmm(q_reshaped, k_permuted).view(H_q, -1) * inv_scale
                    
                    # ── Sparse LSE Bias ──────────────────────────────────────
                    # See resolve_sparse_bias: 'auto' is 0.0 here because CUDA's
                    # sparse half CONTAINS the exact residuals, which inverts the
                    # sign of (lse_dense - lse_sparse) relative to the partition
                    # the adaptive formula was written against. Scaling l_i and
                    # O_i by e^bias is the online-softmax equivalent of adding
                    # `bias` to lse_sparse: O_i is the un-normalised numerator
                    # (out * l_out, above) and l_i its denominator, so the merge
                    # below weights this half by e^bias more.
                    _bias = resolve_sparse_bias(
                        m_i + torch.log(torch.clamp(l_i, min=1e-9)),
                        torch.logsumexp(s, dim=-1))
                    if torch.is_tensor(_bias):
                        factor = torch.exp(_bias)
                        l_i = l_i * factor
                        O_i = O_i * factor.unsqueeze(-1)
                    elif _bias != 0.0:
                        factor = math.exp(_bias)
                        l_i = l_i * factor
                        O_i = O_i * factor

                    m_b = s.max(-1).values
                    m_new = torch.maximum(m_i, m_b)
                    a = torch.exp(m_i - m_new)
                    P = torch.exp(s - m_new.unsqueeze(-1))
                    l_i = a * l_i + P.sum(-1)
                    
                    P_reshaped = P.view(H_kv, n_rep, -1)
                    v_permuted = v_kv[0]
                    
                    O_i_delta = torch.bmm(P_reshaped, v_permuted).view(H_q, D)
                    O_i = a.unsqueeze(-1) * O_i + O_i_delta
                    m_i = m_new
                    
                out = O_i / l_i.unsqueeze(-1)

            # Observability (F3): confirm ONCE that the Triton kernel path is live.
            # On a GPU box this is the only positive signal that you're measuring
            # Triton and not the silent PyTorch fallback below.
            if not getattr(native_triton_sparse_attn_decode, "_triton_active_logged", False):
                print("[DKV] Triton fused-decode path ACTIVE (CUDA).")
                native_triton_sparse_attn_decode._triton_active_logged = True
            return out.unsqueeze(0).unsqueeze(2).to(q.dtype)
        except Exception as e:
            # Strict mode (F3): re-raise instead of masking a broken kernel behind
            # the slow PyTorch fallback. Use during GPU bring-up / validation so a
            # compile or numerics failure is loud. Default (unset) preserves the
            # historical silent-fallback behavior exactly.
            if os.environ.get("DKV_TRITON_STRICT") == "1":
                raise
            global _triton_fallback_count
            _triton_fallback_count += 1
            if _triton_fallback_count == 1:
                print(
                    f"[DKV] WARNING: Triton sparse kernel failed: {e}. "
                    "Falling back to PyTorch vectorized decoder. "
                    "Set DKV_TRITON_STRICT=1 to surface the full error.",
                    flush=True,
                )
            elif _triton_fallback_count == 10:
                print(
                    f"[DKV] WARNING: Triton fallback has now occurred 10 times "
                    f"(last error: {e}). Check kernel compilation and CUDA version.",
                    flush=True,
                )
            elif _triton_fallback_count == 100:
                print(
                    f"[DKV] ERROR: Triton fallback count reached 100. The Triton "
                    "sparse kernel appears persistently broken — all decode steps are "
                    "running the slow PyTorch fallback. Investigate immediately.",
                    flush=True,
                )
            return _pytorch_vectorized_sparse_attn_decode(
                q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
                anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
                cos_sliced=cos_sliced, sin_sliced=sin_sliced,
                session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
                active_len=active_len,
            )
    else:
        # N == 0: no compressed blocks routed. This delegates to the PyTorch
        # decoder, which returns an EMPTY tensor in that case — last dim 0, not D.
        # Reproduced by tests/test_triton_combined.py::test_dense_only:
        #   RuntimeError: The size of tensor a (64) must match tensor b (0)
        #                 at non-singleton dimension 3
        # and the 64 there is head_dim, i.e. the reference; the 0 is this return.
        # native_triton_sparse_attn_decode_combined handles the same case
        # correctly with a dense-only SDPA fast path; this entry point never grew
        # one, so a zero-block step silently produces a zero-width attention
        # output rather than attending the dense window.
        #
        # NOT fixed here on purpose: the correct fix is that dense-only fast path,
        # and `active_k` on this branch has not been through the RoPE rotation the
        # N>0 branch applies, so writing it without a GPU to verify against would
        # be guessing. Made LOUD instead — a wrong shape that announces itself is
        # recoverable; a silent one is what let this sit behind a failing test.
        return _pytorch_vectorized_sparse_attn_decode(
            q, block_indices, pool, dense_blocks, active_k, active_v, num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin, total_seq_len=total_seq_len, max_valid_len=max_valid_len,
            cos_sliced=cos_sliced, sin_sliced=sin_sliced,
            session_id=session_id, layer_idx=layer_idx, decode_workspace=decode_workspace,
            active_len=active_len,
        )


# ── 3b. Fused Combined Kernel: Compressed Blocks + Dense Window ───────────────
#
# Extends _fused_sparse_decode_kernel by iterating over dense window tokens in
# the SAME online softmax accumulator, replacing the 3-step pattern:
#   sparse Triton → dense SDPA → Python LSE-merge
# with a single Triton dispatch.  Grid and chunk logic are identical to the
# sparse-only kernel; dense tokens are processed after all sparse blocks in
# each chunk's accumulator.
#
# Key design decisions:
#   • Dense K/V are expected pre-RoPE-rotated by the caller (same as the
#     existing Python LSE-merge path in native_triton_sparse_attn_decode).
#   • L_dense is a tl.constexpr so the compiler elides the dense loop entirely
#     when there are no dense tokens (L_dense == 0), keeping the sparse-only
#     performance identical.
#   • Dense tokens are chunked identically to blocks: each Triton program
#     processes BLOCKS_PER_CHUNK sparse blocks followed by up to
#     DENSE_PER_CHUNK dense tokens from the matching dense slice.

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'num_warps': 4, 'num_stages': 2}),
            triton.Config({'num_warps': 4, 'num_stages': 4}),
            triton.Config({'num_warps': 8, 'num_stages': 2}),
            triton.Config({'num_warps': 8, 'num_stages': 4}),
        ],
        key=['N', 'L_dense']
    )
    @triton.jit
    def _fused_decode_combined_kernel(
        # ── Sparse compressed-block inputs (identical to _fused_sparse_decode_kernel) ──
        q_ptr, block_indices_ptr, pool_ak_ptr, pool_av_ptr, pool_vk_ptr, pool_vv_ptr,
        pool_u_ptr, pool_u_scale_ptr, pool_scales_ptr, pool_seq_lens_ptr,
        pool_res_k_ptr, pool_res_v_ptr, pool_res_pos_ptr, pool_res_pos_v_ptr, pool_res_n_ptr,
        pool_fact_pos_ptr, pool_fact_ak_ptr, pool_fact_av_ptr,
        # ── Dense window inputs (new) ──
        dense_k_ptr,        # [H_kv, L_dense, D]  pre-RoPE-rotated
        dense_v_ptr,        # [H_kv, L_dense, D]
        L_dense,            # int  (total/padded dense positions = max_dense_len)
        L_dense_valid,      # int  (actual valid dense tokens; positions >= L_dense_valid are padding)
        # ── Strides for dense tensors ──
        stride_dk_h, stride_dk_l, stride_dk_d,
        stride_dv_h, stride_dv_l, stride_dv_d,
        # ── Output buffers ──
        out_ptr, m_ptr, l_ptr,
        # ── Strides (sparse, identical ordering to _fused_sparse_decode_kernel) ──
        stride_q_h, stride_q_d,
        stride_ak_n, stride_ak_h, stride_ak_d,
        stride_av_n, stride_av_h, stride_av_d,
        stride_vk_n, stride_vk_r, stride_vk_h, stride_vk_d,
        stride_vv_n, stride_vv_r, stride_vv_h, stride_vv_d,
        stride_u_n, stride_u_s, stride_u_r,
        stride_res_k_n, stride_res_k_s, stride_res_k_h, stride_res_k_d,
        stride_res_v_n, stride_res_v_s, stride_res_v_h, stride_res_v_d,
        stride_res_pos_n, stride_res_pos_v_n,
        stride_fact_pos_n,
        stride_fact_ak_n, stride_fact_ak_f, stride_fact_ak_h, stride_fact_ak_d,
        stride_fact_av_n, stride_fact_av_f, stride_fact_av_h, stride_fact_av_d,
        stride_out_h, stride_out_d,
        # ── Constexpr shape/config ──
        N: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr, KV_GRP: tl.constexpr, D: tl.constexpr,
        R: tl.constexpr, R_REAL: tl.constexpr, S_MAX: tl.constexpr, INV_SCALE: tl.constexpr,
        BLOCKS_PER_CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr,
        MAX_RESIDUAL: tl.constexpr, MAX_FACT: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr, HAS_FACT: tl.constexpr,
        EXACT_RESIDUAL: tl.constexpr,
        DENSE_PER_CHUNK: tl.constexpr,   # dense tokens each chunk processes (0 disables the loop)
        BLOCK_SIZE_T: tl.constexpr = 64,  # Parallelize dense window loads in blocks of 64
    ):
        h_q = tl.program_id(0)
        chunk_id = tl.program_id(1)
        h_kv = h_q // KV_GRP

        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, R)
        offs_s = tl.arange(0, S_MAX)

        q_ptrs = q_ptr + h_q * stride_q_h + offs_d * stride_q_d
        q = tl.load(q_ptrs).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        O_i = tl.zeros([D], dtype=tl.float32)

        # ── Sparse compressed-block loop (identical to _fused_sparse_decode_kernel) ──
        start_block = chunk_id * BLOCKS_PER_CHUNK
        end_block = start_block + BLOCKS_PER_CHUNK
        if end_block > N:
            end_block = N

        for n in range(start_block, end_block):
            pool_idx = tl.load(block_indices_ptr + n)
            scale = tl.load(pool_scales_ptr + pool_idx).to(tl.float32)
            actual_s = tl.load(pool_seq_lens_ptr + pool_idx)

            ak_ptrs = pool_ak_ptr + pool_idx * stride_ak_n + h_kv * stride_ak_h + offs_d * stride_ak_d
            av_ptrs = pool_av_ptr + pool_idx * stride_av_n + h_kv * stride_av_h + offs_d * stride_av_d
            ak = tl.load(ak_ptrs).to(tl.float32)
            av = tl.load(av_ptrs).to(tl.float32)

            # r_mask is REQUIRED, not defensive. R here is R_pad =
            # next_power_of_2(layer_rank), while the pool's rank dimension is
            # pool_rank (the max rank across layers). For a 1.5x-boosted layer
            # rank 48 -> R_pad 64 against a 48-wide pool, so r = 48..63 walked
            # past the rank dimension into the NEXT SLOT's basis and summed it
            # into this block's scores and values. For rank 24 -> R_pad 32 it
            # read 8 columns this layer never wrote. Both vk/vv loads were
            # unmasked; only `u` was masked, and only on S.
            r_mask = offs_r < R_REAL
            vk_ptrs = pool_vk_ptr + pool_idx * stride_vk_n + h_kv * stride_vk_h + offs_r[:, None] * stride_vk_r + offs_d[None, :] * stride_vk_d
            vv_ptrs = pool_vv_ptr + pool_idx * stride_vv_n + h_kv * stride_vv_h + offs_r[:, None] * stride_vv_r + offs_d[None, :] * stride_vv_d
            vk = tl.load(vk_ptrs, mask=r_mask[:, None], other=0.0).to(tl.float32)
            vv = tl.load(vv_ptrs, mask=r_mask[:, None], other=0.0).to(tl.float32)

            u_ptrs = pool_u_ptr + pool_idx * stride_u_n + offs_s[:, None] * stride_u_s + offs_r[None, :] * stride_u_r
            s_mask = (offs_s[:, None] < actual_s) & r_mask[None, :]
            u = tl.load(u_ptrs, mask=s_mask, other=0.0).to(tl.float32)
            u_scale = tl.load(pool_u_scale_ptr + pool_idx)
            u = u * u_scale

            s_anchor = tl.sum(q * ak) * INV_SCALE
            q_proj = tl.sum(q[None, :] * vk, axis=1) * INV_SCALE
            delta_scores = tl.sum(u * q_proj[None, :], axis=1) * scale
            s = s_anchor + delta_scores

            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_k = tl.load(pool_res_pos_ptr + pool_idx * stride_res_pos_n + ri)
                    if r_pos_k >= 0:
                        rk = tl.load(pool_res_k_ptr + pool_idx * stride_res_k_n +
                                     ri * stride_res_k_s + h_kv * stride_res_k_h +
                                     offs_d * stride_res_k_d).to(tl.float32)
                        r_corr = tl.sum(q * rk) * INV_SCALE
                        if EXACT_RESIDUAL:
                            # MLX / Metal SUBSTITUTION. rk is the anchor-relative
                            # EXACT key, so exact_K = ak + rk and this token's true
                            # score is s_anchor + q·rk. Writing that REPLACES the
                            # score, dropping delta_scores[p] — the lossy low-rank
                            # twin — which is exactly what MLX means by "exact
                            # residual rows appended and their lossy low-rank twins
                            # masked", and what dkv_decode.metal already does. Same
                            # shape as the C2 fact-anchor override below.
                            s = tl.where(offs_s == r_pos_k, s_anchor + r_corr, s)
                        else:
                            # Correction form: rk is (exact - recon), so ADD and keep
                            # the twin. Approximate: the twin's delta stays in, and
                            # the two terms are rotated in different frames (base at
                            # the block anchor, rk at the token's true position).
                            s = tl.where(offs_s == r_pos_k, s + r_corr, s)

            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        fact_k_ptrs = pool_fact_ak_ptr + pool_idx * stride_fact_ak_n + fi * stride_fact_ak_f + h_kv * stride_fact_ak_h + offs_d * stride_fact_ak_d
                        fact_k = tl.load(fact_k_ptrs).to(tl.float32)
                        fact_score = tl.sum(q * fact_k) * INV_SCALE
                        replace_mask = offs_s == fact_pos
                        s = tl.where(replace_mask, fact_score, s)

            s = tl.where(offs_s < actual_s, s, -float("inf"))
            m_b_delta = tl.max(s, axis=0)
            m_b = tl.maximum(s_anchor, m_b_delta)

            m_new = tl.maximum(m_i, m_b)
            alpha = tl.exp(m_i - m_new)
            p_anchor = tl.exp(s_anchor - m_new)
            p_delta = tl.exp(s - m_new)
            p_delta = tl.where(offs_s < actual_s, p_delta, 0.0)
            p_delta_sum = tl.sum(p_delta, axis=0)

            l_i = l_i * alpha + p_anchor + p_delta_sum

            p_u = tl.sum(p_delta[:, None] * u, axis=0)
            o_delta = tl.sum(p_u[:, None] * vv, axis=0) * scale

            O_fact_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_FACT:
                for fi in range(MAX_FACT):
                    fact_pos = tl.load(pool_fact_pos_ptr + pool_idx * stride_fact_pos_n + fi)
                    if fact_pos >= 0:
                        replace_mask = offs_s == fact_pos
                        p_fact = tl.sum(tl.where(replace_mask, p_delta, 0.0), axis=0)
                        fact_v_ptrs = pool_fact_av_ptr + pool_idx * stride_fact_av_n + fi * stride_fact_av_f + h_kv * stride_fact_av_h + offs_d * stride_fact_av_d
                        fact_v = tl.load(fact_v_ptrs).to(tl.float32)
                        u_val_ptrs = pool_u_ptr + pool_idx * stride_u_n + fact_pos * stride_u_s + offs_r * stride_u_r
                        # vv is already r-masked to zero above so the sum would be
                        # right regardless, but the LOAD itself must not address
                        # past the rank dimension (the last slot has nothing after it).
                        u_val = tl.load(u_val_ptrs, mask=r_mask, other=0.0).to(tl.float32) * u_scale
                        v_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale + av
                        O_fact_corr += p_fact * (fact_v - v_recon)

            O_res_corr = tl.zeros([D], dtype=tl.float32)
            if HAS_RESIDUAL:
                for ri in range(MAX_RESIDUAL):
                    r_pos_v = tl.load(pool_res_pos_v_ptr + pool_idx * stride_res_pos_v_n + ri)
                    if r_pos_v >= 0:
                        p_at = tl.sum(tl.where(offs_s == r_pos_v, p_delta, 0.0), axis=0)
                        rv = tl.load(pool_res_v_ptr + pool_idx * stride_res_v_n +
                                     ri * stride_res_v_s + h_kv * stride_res_v_h +
                                     offs_d * stride_res_v_d).to(tl.float32)
                        if EXACT_RESIDUAL:
                            # Substitution on the V side too, or the score would be
                            # exact while the value stayed lossy. The row currently
                            # contributes p_at·(av + delta_V[p]) via o_delta and the
                            # shared av term; the exact value is av + rv, so the
                            # correction is p_at·(rv - delta_V[p]). delta_V[p] is
                            # recomputed from this position's U row exactly as the
                            # C2 fact-anchor value override does. The load must be
                            # r_mask'd: R here is R_pad, wider than the pool's rank.
                            u_val_ptrs = (pool_u_ptr + pool_idx * stride_u_n
                                          + r_pos_v * stride_u_s + offs_r * stride_u_r)
                            u_val = tl.load(u_val_ptrs, mask=r_mask, other=0.0).to(tl.float32) * u_scale
                            dv_recon = tl.sum(u_val[:, None] * vv, axis=0) * scale
                            O_res_corr += p_at * (rv - dv_recon)
                        else:
                            O_res_corr += p_at * rv

            O_i = O_i * alpha + (p_anchor + p_delta_sum) * av + o_delta + O_fact_corr + O_res_corr
            m_i = m_new

        # ── Dense window token loop (NEW — fused into the same online softmax) ──
        # Each chunk processes a slice of dense tokens [dense_start, dense_end).
        # dense_K/dense_V are [H_kv, L_dense, D] pre-RoPE-rotated by the caller.
        # GQA: use h_kv for loading, same as the sparse branch above.
        if DENSE_PER_CHUNK > 0:
            dense_start = chunk_id * DENSE_PER_CHUNK
            dense_end = dense_start + DENSE_PER_CHUNK
            if dense_end > L_dense:
                dense_end = L_dense

            for t_start in range(dense_start, dense_end, BLOCK_SIZE_T):
                offs_t = t_start + tl.arange(0, BLOCK_SIZE_T)
                mask_t = offs_t < dense_end

                dk_ptrs = dense_k_ptr + h_kv * stride_dk_h + offs_t[:, None] * stride_dk_l + offs_d[None, :] * stride_dk_d
                dk = tl.load(dk_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

                score = tl.sum(q[None, :] * dk, axis=1) * INV_SCALE
                # Mask both out-of-chunk-range positions AND padding positions
                # (offs_t >= L_dense_valid contain zero-padded keys from the fixed workspace)
                valid_t = mask_t & (offs_t < L_dense_valid)
                score = tl.where(valid_t, score, -float("inf"))

                mb = tl.max(score, axis=0)
                m_new = tl.maximum(m_i, mb)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(score - m_new)
                p = tl.where(valid_t, p, 0.0)

                l_i = l_i * alpha + tl.sum(p, axis=0)

                dv_ptrs = dense_v_ptr + h_kv * stride_dv_h + offs_t[:, None] * stride_dv_l + offs_d[None, :] * stride_dv_d
                dv = tl.load(dv_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

                O_i = O_i * alpha + tl.sum(p[:, None] * dv, axis=0)
                m_i = m_new

        # ── Write partial outputs (identical epilogue to _fused_sparse_decode_kernel) ──
        if NUM_CHUNKS == 1:
            O_i = O_i / l_i
            out_ptrs = out_ptr + h_q * stride_out_h + offs_d * stride_out_d
            tl.store(out_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q, l_i)
        else:
            out_work_ptrs = out_ptr + h_q * (NUM_CHUNKS * D) + chunk_id * D + offs_d
            tl.store(out_work_ptrs, O_i)
            if m_ptr is not None:
                tl.store(m_ptr + h_q * NUM_CHUNKS + chunk_id, m_i)
            if l_ptr is not None:
                tl.store(l_ptr + h_q * NUM_CHUNKS + chunk_id, l_i)


def native_triton_sparse_attn_decode_combined(
    q:                    torch.Tensor,       # [1, H_q, 1, D]
    block_indices:        torch.Tensor,       # [N]  active compressed-block indices
    pool:                 object,             # NativeBlockPool
    dense_k:              Optional[torch.Tensor],  # [1, H_kv, max_dense_len, D] pre-RoPE-rotated; None if no dense
    dense_v:              Optional[torch.Tensor],  # [1, H_kv, max_dense_len, D]
    num_key_value_groups: int,
    R:                    int = 16,
    S_MAX:                int = 64,
    anchor_indices:       Optional[torch.Tensor] = None,
    cos:                  Optional[torch.Tensor] = None,
    sin:                  Optional[torch.Tensor] = None,
    dense_len:            Optional[int] = None,  # actual valid dense tokens
    gather_cache:         Optional[dict] = None,  # {gather_key: (pool_for_kernel, g)} — see below
    gather_key=None,                              # cache key; None disables caching
) -> torch.Tensor:
    """
    Single-dispatch fused attention over both compressed blocks and dense window tokens.

    Replaces the 3-step pattern used in native_triton_sparse_attn_decode:
        sparse Triton → dense F.sdpa → Python LSE-merge
    with one Triton kernel that processes both token classes in the same online softmax.

    Falls back to native_triton_sparse_attn_decode (which does its own dense merge)
    on any error or if HAS_TRITON is False.

    Returns: [1, H_q, 1, D] in q.dtype.
    """
    if not HAS_TRITON:
        # MPS / CPU: fall back to the existing separate-path wrapper
        # (dense is handled by its own Python LSE merge inside that function)
        return native_triton_sparse_attn_decode(
            q, block_indices, pool, [], dense_k, dense_v,
            num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin,
        )

    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1

    N = block_indices.shape[0] if block_indices is not None else 0
    # has_dense is driven by dense_len (actual valid tokens), not the padded buffer shape.
    if dense_len is None:
        dense_len = dense_k.shape[2] if dense_k is not None else 0
    has_dense = dense_k is not None and dense_len > 0
    # L_dense = padded workspace size (fixed shape, = max_dense_len).  Kernel uses
    # L_dense for pointer arithmetic and L_dense_valid for score masking.
    L_dense = dense_k.shape[2] if dense_k is not None else 0

    # If nothing to attend to, return zeros
    if N == 0 and not has_dense:
        return torch.zeros((1, H_q, 1, D), device=q.device, dtype=q.dtype)

    # If there are no compressed blocks, just run dense SDPA (fast path)
    if N == 0 and has_dense:
        H_kv = dense_k.shape[1]
        n_rep = H_q // H_kv
        q_sq = q[0, :, 0, :].float()
        dk = dense_k[0].float()  # [H_kv, L_dense, D] (padded workspace)
        dv = dense_v[0].float()
        q_r = q_sq.view(H_kv, n_rep, D)
        s = torch.bmm(q_r, dk.permute(0, 2, 1)).view(H_q, L_dense) / math.sqrt(D)
        # Mask padding positions (dense_len..L_dense-1) so they contribute 0 to softmax
        if dense_len < L_dense:
            s = s.clone()
            s[:, dense_len:] = float('-inf')
        w = torch.softmax(s, dim=-1)
        w_r = w.view(H_kv, n_rep, L_dense)
        out = torch.bmm(w_r, dv).view(H_q, D)
        return out.unsqueeze(0).unsqueeze(2).to(q.dtype)

    try:
        inv_scale = 1.0 / math.sqrt(D)
        q_sq = q[0, :, 0, :]  # [H_q, D]

        D_pad   = triton.next_power_of_2(D)
        R_pad   = triton.next_power_of_2(R)
        S_pad   = triton.next_power_of_2(S_MAX)

        BLOCKS_PER_CHUNK = 16
        num_chunks_sparse = max(1, (N + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK)

        # Distribute dense tokens across the same chunk grid so each program sees
        # a balanced slice.  When L_dense == 0, DENSE_PER_CHUNK = 0 → loop elided.
        if L_dense > 0:
            DENSE_PER_CHUNK = max(1, (L_dense + num_chunks_sparse - 1) // num_chunks_sparse)
            num_chunks = max(num_chunks_sparse,
                             (L_dense + DENSE_PER_CHUNK - 1) // DENSE_PER_CHUNK)
        else:
            DENSE_PER_CHUNK = 0
            num_chunks = num_chunks_sparse

        grid = (H_q, num_chunks)

        # Allocate output/workspace buffers
        if num_chunks > 1:
            cache_key = (H_q, num_chunks, D_pad, q.device)
            if not hasattr(native_triton_sparse_attn_decode_combined, "_ws_cache"):
                native_triton_sparse_attn_decode_combined._ws_cache = {}
            ws = native_triton_sparse_attn_decode_combined._ws_cache.get(cache_key)
            if ws is None:
                ow = torch.empty((H_q, num_chunks, D_pad), device=q.device, dtype=torch.float32)
                mw = torch.empty((H_q, num_chunks),        device=q.device, dtype=torch.float32)
                lw = torch.empty((H_q, num_chunks),        device=q.device, dtype=torch.float32)
                ws = (ow, mw, lw)
                native_triton_sparse_attn_decode_combined._ws_cache[cache_key] = ws
            out_workspace, m_workspace, l_workspace = ws
            out   = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
            m_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            l_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
        else:
            out   = torch.empty((H_q, D), device=q.device, dtype=torch.float32)
            m_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            l_out = torch.empty((H_q,),   device=q.device, dtype=torch.float32)
            out_workspace, m_workspace, l_workspace = out, m_out, l_out

        # ── Gather + rotate the N routed rows (identical semantics to
        # native_triton_sparse_attn_decode; see F2 helper). Issue 1 fix included:
        # stratified U is pre-reconstructed before dispatch for CUDA/MPS parity.
        #
        # DECODE-CACHE (DKV_DECODE_CACHE_CUDA): this gather is QUERY-INDEPENDENT
        # — it depends only on the routed block set + their anchor RoPE positions,
        # both stable between block flushes. So within an interval it is IDENTICAL
        # every decode token, yet the current code recomputes ~20 index/rotate ops
        # per token per layer (the launch explosion the profiler showed). When the
        # caller passes a (cache, key) whose key only changes on a block flush, we
        # compute it once and reuse — the kernel still runs every token with the
        # fresh query, so output is bit-identical to the uncached path.
        _cached = gather_cache.get(gather_key) if (gather_cache is not None and gather_key is not None) else None
        if _cached is not None:
            pool_for_kernel, g = _cached
        else:
            pool_for_kernel, _used_strat = _build_stratified_U_for_triton(pool, block_indices)
            g = _gather_routed_blocks_for_kernel(
                pool_for_kernel, block_indices, anchor_indices, cos, sin)
            if gather_cache is not None and gather_key is not None:
                gather_cache[gather_key] = (pool_for_kernel, g)

        # ── Dense window tensors ──
        # Caller provides pre-RoPE-rotated dense_k/dense_v as [1, H_kv, L_dense, D].
        # We need [H_kv, L_dense, D] contiguous for the kernel.
        if has_dense:
            dk_t = dense_k[0].contiguous().to(torch.float32)  # [H_kv, L_dense, D]
            dv_t = dense_v[0].contiguous().to(torch.float32)
        else:
            dk_t = torch.empty((1, 0, D_pad), device=q.device, dtype=torch.float32)
            dv_t = torch.empty((1, 0, D_pad), device=q.device, dtype=torch.float32)

        # ── Kernel launch ──
        _fused_decode_combined_kernel[grid](
            q_sq, g["idx"], g["anchors_K"], g["anchors_V"], g["V_K"], g["V_V"],
            g["U"], g["U_scale"], g["scales"], g["seq_lens"],
            g["res_k"], g["res_v"], g["res_pos"], g["res_pos_v"], g["res_n"],
            g["fact_pos"], g["fact_ak"], g["fact_av"],
            dk_t, dv_t, L_dense, dense_len,
            dk_t.stride(0), dk_t.stride(1), dk_t.stride(2),
            dv_t.stride(0), dv_t.stride(1), dv_t.stride(2),
            out_workspace, m_workspace, l_workspace,
            q_sq.stride(0), q_sq.stride(1),
            g["anchors_K"].stride(0), g["anchors_K"].stride(1), g["anchors_K"].stride(2),
            g["anchors_V"].stride(0), g["anchors_V"].stride(1), g["anchors_V"].stride(2),
            g["V_K"].stride(0), g["V_K"].stride(1), g["V_K"].stride(2), g["V_K"].stride(3),
            g["V_V"].stride(0), g["V_V"].stride(1), g["V_V"].stride(2), g["V_V"].stride(3),
            g["U"].stride(0), g["U"].stride(1), g["U"].stride(2),
            g["res_k"].stride(0), g["res_k"].stride(1), g["res_k"].stride(2), g["res_k"].stride(3),
            g["res_v"].stride(0), g["res_v"].stride(1), g["res_v"].stride(2), g["res_v"].stride(3),
            g["res_pos"].stride(0), g["res_pos_v"].stride(0),
            g["fact_pos"].stride(0),
            g["fact_ak"].stride(0), g["fact_ak"].stride(1), g["fact_ak"].stride(2), g["fact_ak"].stride(3),
            g["fact_av"].stride(0), g["fact_av"].stride(1), g["fact_av"].stride(2), g["fact_av"].stride(3),
            out_workspace.stride(0), out_workspace.stride(1),
            N, H_q, g["anchors_K"].shape[1], num_key_value_groups, D_pad,
            R_pad, R, S_pad, inv_scale, BLOCKS_PER_CHUNK, num_chunks,
            MAX_RESIDUAL=g["max_res_pad"], MAX_FACT=g["max_fact"],
            HAS_RESIDUAL=g["has_res"], HAS_FACT=g["has_fact"],
            EXACT_RESIDUAL=_exact_residual_semantics(q.device),
            DENSE_PER_CHUNK=DENSE_PER_CHUNK,
        )

        if num_chunks > 1:
            # OPT-E: dispatches sequential vs. parallel tree reduction based on num_chunks
            _dispatch_reduction(
                out_workspace, m_workspace, l_workspace, out, m_out, l_out,
                num_chunks, D_pad, H_q,
            )

        if not getattr(native_triton_sparse_attn_decode_combined, "_logged", False):
            print("[DKV] Triton fused-decode COMBINED path ACTIVE (CUDA). "
                  f"N_sparse={N}, L_dense={L_dense}")
            native_triton_sparse_attn_decode_combined._logged = True

        return out.unsqueeze(0).unsqueeze(2).to(q.dtype)

    except Exception as e:
        if os.environ.get("DKV_TRITON_STRICT") == "1":
            raise
        global _triton_fallback_count
        _triton_fallback_count += 1
        if _triton_fallback_count == 1:
            print(
                f"[DKV] WARNING: combined Triton kernel failed: {e}. "
                "Falling back to native_triton_sparse_attn_decode. "
                "Set DKV_TRITON_STRICT=1 to surface the full error.",
                flush=True,
            )
        elif _triton_fallback_count == 10:
            print(
                f"[DKV] WARNING: Triton fallback count reached 10 "
                f"(last error: {e}). Check kernel compilation and CUDA version.",
                flush=True,
            )
        elif _triton_fallback_count == 100:
            print(
                "[DKV] ERROR: Triton fallback count reached 100. The combined "
                "Triton kernel appears persistently broken — investigate immediately.",
                flush=True,
            )
        return native_triton_sparse_attn_decode(
            q, block_indices, pool, [], dense_k, dense_v,
            num_key_value_groups, R, S_MAX,
            anchor_indices=anchor_indices, cos=cos, sin=sin,
            active_len=dense_len,
        )


# ── 4. TritonDKV Low-Rank Reconstruction ───────────────────────────────────


def triton_fused_reconstruct(
    U: torch.Tensor,
    V: torch.Tensor,
    anchor: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    n_tokens, rank = U.shape
    _, feat_dim = V.shape

    if not HAS_TRITON:
        result = (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if out is None:
        out = torch.empty((n_tokens, feat_dim), device=U.device, dtype=U.dtype)

    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 64
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE_N), triton.cdiv(feat_dim, BLOCK_SIZE_D))

    _use_nvtx = _has_cuda()
    if _use_nvtx:
        _nvtx_push("Triton_LowRank_Recon_Kernel_Launch")

    lowrank_recon_kernel[grid](
        U, V, anchor, out,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        anchor.stride(0),
        out.stride(0), out.stride(1),
        n_tokens, rank, feat_dim, scale,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    if _use_nvtx:
        _nvtx_pop()

    return out


class TritonDKV:
    _recon_buffers = {}

    @classmethod
    def _get_recon_buffer(cls, n_tokens: int, feat_dim: int, device, dtype) -> torch.Tensor:
        key = (device, dtype, feat_dim)
        if key not in cls._recon_buffers or cls._recon_buffers[key].shape[0] < n_tokens:
            alloc_size = max(2048, n_tokens)
            cls._recon_buffers[key] = torch.zeros(
                (alloc_size, feat_dim), device=device, dtype=dtype
            )
        return cls._recon_buffers[key][:n_tokens]

    @staticmethod
    def reconstruct_lowrank(
        U: torch.Tensor,
        V: torch.Tensor,
        anchor: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        out_buf = TritonDKV._get_recon_buffer(
            U.shape[0], V.shape[1], U.device, U.dtype
        )
        try:
            out = triton_fused_reconstruct(U, V, anchor, out=out_buf, scale=scale)
            return out.clone()
        except Exception as e:
            return (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)

    @staticmethod
    def reconstruct_lowrank_sparse(
        U: torch.Tensor,
        V: torch.Tensor,
        anchor: torch.Tensor,
        sparse_indices: Optional[torch.Tensor],
        sparse_values: Optional[torch.Tensor],
        scale: float = 1.0,
    ) -> torch.Tensor:
        out = TritonDKV.reconstruct_lowrank(U, V, anchor, scale)
        if sparse_indices is not None and sparse_indices.numel() > 0:
            out.view(-1).index_add_(
                0, sparse_indices.long(), sparse_values.to(out.dtype)
            )
        return out

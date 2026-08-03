import sys
import os
# Add the build directory containing dkv_core.so to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.abspath(os.path.join(_script_dir, "../native_core/dkv_core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

if os.environ.get("DKV_FORCE_PYTORCH") == "1" and sys.platform != "darwin":
    sys.modules["dkv_core"] = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import threading
from typing import Optional, Tuple, Dict
from native_core.sparse_decode.triton_fused_decode import (
    TritonDKV,
    native_triton_sparse_attn_decode,
    native_triton_sparse_attn_decode_combined,
    _prefill_fused_history_attend,
    fused_decode_mps,
    _DKV_DEBUG_NUMERICS,
    HAS_TRITON,
    resolve_sparse_bias as _resolve_sparse_bias,
    pool_stores_rotated_k as _pool_rotated_k,
    _exact_residual_semantics,
)
from native_core.compression.lowrank import reconstruct_batch_U


def _ingest_k(rot_k, unrot_k):
    """The K that PREFILL stores must match what the decoder assumes it stored.

    `pool_stores_rotated_k()` tells the decode gather whether to re-rotate
    (`do_rot = ... and not pool_stores_rotated_k()`). Every prefill capture site
    passed `unrot_key_states` UNCONDITIONALLY while that predicate returned True,
    so the pool held PRE-RoPE keys and the decoder skipped the rotation they
    needed -- RoPE was simply absent from all compressed content.

    Measured, not inferred: probe_residual_values scored anchor+residual against
    ground truth and got cos_RAW = 1.0000 (bit exact, unrotated) versus
    cos_ROT = 0.84-0.98, with |K_pool| == |K_true| at every layer because RoPE is
    orthogonal. It also explains the depth gradient -- at depth 0.0 the block sits
    near position 0 where RoPE is ~identity, so it passed while deeper needles
    degraded.

    Routing through one helper means the two sides cannot silently disagree again.
    """
    return rot_k if _pool_rotated_k() else unrot_k

# ── Phase 1: C++ extension fast path ─────────────────────────────────────────
# Import dkv_core C++ extension if built. When available, the hot path uses:
#   - dkv_core.anchor_screen()          instead of Python two_level_gate()
#   - dkv_core.semantic_search_topk()   instead of SemanticIndex.search()
#   - dkv_core.compute_query_desc()     instead of compute_query_descriptor()
#   - dkv_core.decode_attention_aten()  instead of fused_decode_mps()
#   - dkv_core.decode_attention_aten_lse() for LSE-combine path
#
# The extension is built by running:
#   cd ACTIVE_RUNTIME/native_core/dkv_core && python setup.py build_ext --inplace
#
# Falls back silently to the existing Python paths if the extension is absent
# or built without the Phase 1 ops. No behavioral change on fallback.
try:
    import dkv_core as _dkv_core

    # ── Stale-binary guard ────────────────────────────────────────────────────
    # A bare `import dkv_core` resolves to whatever is first on sys.path, and
    # there are usually SEVERAL candidates: a copy at the repo root, the real
    # build in native_core/dkv_core/, and two editable-install .pth finders
    # (__editable__.dkv_core-*.pth). Rebuilding in native_core/dkv_core/ does NOT
    # update the root copy, so the runtime can keep running a months-old kernel
    # while the isolated tests -- which insert the build directory FIRST -- happily
    # exercise the new one. Every end-to-end measurement then describes code that
    # is not the code under test, with nothing in the logs to say so.
    #
    # Compare the loaded binary against the Metal source it is built from and say
    # so loudly. Cheap: two stat() calls at import.
    try:
        import os as _os
        _so_path = getattr(_dkv_core, "__file__", None)
        _metal_src = _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), "native_core", "dkv_core", "metal", "dkv_decode.metal")
        if _so_path and _os.path.exists(_metal_src):
            _so_mt, _src_mt = _os.path.getmtime(_so_path), _os.path.getmtime(_metal_src)
            if _src_mt > _so_mt:
                print(f"[DKV] WARNING: loaded dkv_core is OLDER than the Metal source it is "
                      f"built from -- you are running a stale kernel.\n"
                      f"       loaded : {_so_path}\n"
                      f"       source : {_metal_src} (newer by {(_src_mt - _so_mt) / 60:.0f} min)\n"
                      f"       Rebuild:  cd ACTIVE_RUNTIME/native_core/dkv_core && "
                      f"rm -rf build dkv_core*.so && python setup.py build_ext --inplace\n"
                      f"       then copy the built .so over the one listed above.", flush=True)
    except Exception:
        pass  # never let the guard break the import

    _DKV_CORE_AVAILABLE    = True
    _DKV_HAS_DECODE_ATTN   = getattr(_dkv_core, "HAS_DECODE_ATTN", False)
    _DKV_HAS_SRL_ROUTER    = getattr(_dkv_core, "HAS_SRL_ROUTER", False)
    _DKV_HAS_METAL_ATTN    = getattr(_dkv_core, "HAS_METAL_ATTN", False)
    if _DKV_HAS_DECODE_ATTN:
        _decode_attention_aten     = _dkv_core.decode_attention_aten
        _decode_attention_aten_lse = _dkv_core.decode_attention_aten_lse
    if _DKV_HAS_METAL_ATTN:
        _decode_attention_metal    = _dkv_core.decode_attention_metal
    if _DKV_HAS_SRL_ROUTER:
        _cpp_anchor_screen         = _dkv_core.anchor_screen
        _cpp_semantic_search       = _dkv_core.semantic_search_topk
        _cpp_query_desc            = _dkv_core.compute_query_desc
except Exception as e:
    print(f"[DKV DEBUG] Failed to import dkv_core: {e}", flush=True)
    import traceback
    traceback.print_exc()
    _DKV_CORE_AVAILABLE    = False
    _DKV_HAS_DECODE_ATTN   = False
    _DKV_HAS_SRL_ROUTER    = False
    _DKV_HAS_METAL_ATTN    = False

# ── SRL routing configuration ─────────────────────────────────────────────────
# DKV_SRL_THRESHOLD: minimum N_blocks before SRL kicks in (default 50).
# DKV_VALIDATE_SRL:  enable accuracy validation mode (0/1, default 0).
# DKV_VALIDATE_EVERY: validate every N decode steps (default 50).
_SRL_THRESHOLD      = int(os.environ.get("DKV_SRL_THRESHOLD",    "50"))
_SRL_VALIDATE       = os.environ.get("DKV_VALIDATE_SRL",         "0") == "1"
_SRL_VALIDATE_EVERY = int(os.environ.get("DKV_VALIDATE_EVERY", "50"))

# ── Decode gather-cache (MLX-parity, DKV_DECODE_CACHE_CUDA) ─────────────────
# The per-token block gather+rotate fed to the combined Triton kernel is
# QUERY-INDEPENDENT — it only changes when the routed block set changes (a block
# flush). Caching it per (session, layer) keyed on the streaming metadata version
# recomputes it once per flush instead of every token, eliminating the launch
# explosion the profiler showed. Output is bit-identical (kernel still runs each
# token with the fresh query). Default OFF until A100-validated (NIAH parity).
#
# NAME COLLISION WARNING: this is NOT the same knob as MLX's `DKV_DECODE_CACHE`
# (mlx_dkv_wrapper.py). MLX's flag changes ROUTING FRESHNESS — it re-routes and
# re-materializes (dequantize + reconstruct) the selected blocks once every
# DKV_DECODE_CACHE_INTERVAL tokens (default 16), trading staleness for speed.
# This CUDA flag never re-routes or re-materializes anything; it only caches
# the gather/index-select step for an UNCHANGED block set between flushes, so
# the routed content is always current — a much narrower optimization. Porting
# one flag's expectations (e.g. an "interval") to the other would be wrong.
_DECODE_CACHE_CUDA = os.environ.get("DKV_DECODE_CACHE_CUDA", "0") == "1"

# ── Re-materialisation cache (DKV_REMAT_CACHE, default OFF) ──────────────────
# Read once at import; see native_core/sparse_decode/remat_cache.py for the
# staleness contract. _LAST_DKV_LAYER tracks which layer index is the last
# DKV-compressed one per manager, so the per-token step counter advances exactly
# once per generated token rather than once per layer.
try:
    from native_core.sparse_decode.remat_cache import remat_enabled as _remat_enabled
    _REMAT_ENABLED = _remat_enabled()
except Exception:                                              # noqa: BLE001
    _REMAT_ENABLED = False
_LAST_DKV_LAYER: dict = {}

# ── Sparse LSE Bias Configuration ─────────────────────────────────────────────
# Parsing lives in resolve_sparse_bias (triton_fused_decode.py), which both merge
# sites now call. The module-level constants that used to sit here were parsed at
# IMPORT time, so any env change after import was silently ignored -- the same
# class of trap as the validator writing DKV_RESIDUAL_EXACT_ROPE after the module
# that reads it was already loaded.

def _apply_sparse_bias(lse_sparse, lse_dense):
    """Delegates to resolve_sparse_bias so this and the inline merge inside
    native_triton_sparse_attn_decode cannot drift -- they were two independent
    copies of the same formula, and the formula turned out to be wrong for
    CUDA's softmax partition. The full argument is in resolve_sparse_bias's
    docstring; the short version is that CUDA keeps the exact residuals in the
    SPARSE half while MLX keeps them in the DENSE half, which flips the sign of
    (lse_dense - lse_sparse) in the needle case and pinned 'auto' at its maximum
    +2.0 nats instead of decaying it to 0."""
    bias = _resolve_sparse_bias(lse_sparse, lse_dense)
    if torch.is_tensor(bias) or bias != 0.0:
        return torch.where(lse_sparse <= -1e9, lse_sparse, lse_sparse + bias)
    return lse_sparse

# ── Context-aware bypass threshold ────────────────────────────────────────────
# DKV_ENGAGE_THRESHOLD: total token count (prefill + history) below which
# DKV bypasses all custom logic and falls through to pure Dense SDPA.
# Rationale: at short contexts there is nothing to compress and no compressed
# history to retrieve — all DKV overhead is pure cost with zero benefit.
#
# Default raised from 2048 → 4096 based on MPS benchmarks:
#   - ≤4K: DKV bypasses to pure dense. Prefill is identical to baseline.
#     Dense handles these contexts fine without memory pressure.
#   - 4K+: DKV engages. Decode is faster (+46% at 4K vs Dense) and VRAM
#     is dramatically lower (-40% at 4K, -96% at 1K bypass mode). At 8K,
#     dense OOMs outright — DKV is the only viable path.
#   - On MPS, synchronous SVD (async disabled for thread safety) adds ~5-7s
#     prefill overhead at 2K (144 SVD ops × 24 layers). Not worth it at 2K.
# Override with DKV_ENGAGE_THRESHOLD=<n> to tune for your hardware.
def _get_engage_threshold():
    return int(os.environ.get("DKV_ENGAGE_THRESHOLD", "4096"))


def _sparse_prefill_filter_blocks(history_blocks, chunk_q, sink_blocks: int = 1,
                                  chunk_start: int = None):
    """DSA/NSA-style block-sparse PREFILL (MLX parity: DKV_SPARSE_PREFILL).

    The "CHUNKED SPARSE PREFILL" path below cross-attends EVERY history block
    for every new chunk -- O(N^2) work that grows with total prompt length,
    despite the section's name. MLX's _sparse_prefill_attend avoids this by
    keeping a few leading "sink" blocks plus only the top-K most
    query-relevant blocks (cheap Quest-style key-bound scoring), dropping
    everything else from that chunk's cross-attention. This mirrors that
    algorithm, adapted for this file's block-OBJECT model (vs MLX's flat
    pool-tensor indexing): score each candidate block by its anchor key's
    dot product with this chunk's (mean-pooled) query -- the same anchor-based
    relevance signal the decode-time router already uses -- and keep only
    sinks + top-K. Vectorized (one stack + one topk), not a per-block .item()
    loop, matching this codebase's stated sync-avoidance discipline elsewhere.

    Gates:
      DKV_SPARSE_PREFILL       default "1" -- set "0" to disable (attend all
                                history blocks, the original behavior).
      DKV_SPARSE_PREFILL_MIN   default 2048 -- MLX parity. Stay fully dense until
                                the chunk starts this far in. Small prompts have
                                nothing worth pruning and the routing overhead
                                does not amortize. This side had NO such gate and
                                sparsified from the very first chunk.
      DKV_SPARSE_PREFILL_WINDOW default 1024 -- MLX parity. Blocks covering the
                                most recent WINDOW tokens are ALWAYS attended, on
                                top of sinks and top-K. This side had no recency
                                guarantee at all, so routing could drop a chunk's
                                immediate left context -- which every token
                                depends on, at every layer.
      DKV_SPARSE_PREFILL_KMIN  default 8   -- minimum routed blocks kept.
      DKV_SPARSE_PREFILL_FRAC  default 0.25 -- routed blocks as a fraction of
                                the routable (non-sink) candidate count. MLX uses
                                0.05; keeping 0.25 here means this side attends
                                strictly MORE than MLX, which is the safe
                                direction, so it is left alone.
    """
    if os.environ.get("DKV_SPARSE_PREFILL", "1") == "0":
        return history_blocks
    if len(history_blocks) <= sink_blocks:
        return history_blocks

    # MLX: `if manager._sparse_prefill and _cur_start >= manager._sp_min_ctx`.
    try:
        min_ctx = int(os.environ.get("DKV_SPARSE_PREFILL_MIN", "2048"))
    except ValueError:
        min_ctx = 2048
    if chunk_start is not None and chunk_start < min_ctx:
        return history_blocks

    try:
        window = int(os.environ.get("DKV_SPARSE_PREFILL_WINDOW", "1024"))
    except ValueError:
        window = 1024

    sinks = history_blocks[:sink_blocks]
    routable = history_blocks[sink_blocks:]

    # Split off the exact recency window: those blocks bypass routing entirely.
    recent = []
    if chunk_start is not None and window > 0 and routable:
        win_start = max(0, int(chunk_start) - window)
        keep_recent = [b for b in routable if getattr(b, "anchor_idx", -1) >= win_start]
        if keep_recent:
            recent_ids = {id(b) for b in keep_recent}
            routable = [b for b in routable if id(b) not in recent_ids]
            recent = keep_recent
    if not routable:
        return sinks + recent

    try:
        kmin = int(os.environ.get("DKV_SPARSE_PREFILL_KMIN", "8"))
    except ValueError:
        kmin = 8
    try:
        frac = float(os.environ.get("DKV_SPARSE_PREFILL_FRAC", "0.25"))
    except ValueError:
        frac = 0.25

    k_eff = min(len(routable), max(kmin, int(math.ceil(frac * len(routable)))))
    if k_eff >= len(routable):
        return history_blocks

    valid = [(i, b) for i, b in enumerate(routable) if getattr(b, "anchor_kv", None) is not None]
    if len(valid) <= k_eff:
        return history_blocks

    device = chunk_q.device
    anchor_ks = torch.stack([b.anchor_kv[0, 0] for _, b in valid], dim=0).float().to(device)  # [nb, H_kv, D]
    q_repr = chunk_q[0].mean(dim=(0, 1)).float()  # [D] -- mean over heads and chunk tokens
    scores = torch.einsum("nhd,d->nh", anchor_ks, q_repr).mean(dim=1)  # [nb], stays on-device
    top_idx = torch.topk(scores, k=k_eff).indices.tolist()  # single sync, not per-block
    keep_positions = sorted(valid[i][0] for i in top_idx)
    kept = sinks + [routable[i] for i in keep_positions] + recent
    # Preserve absolute order: downstream builds positions from anchor_idx and
    # assumes the block list is monotonically ordered.
    kept.sort(key=lambda b: getattr(b, "anchor_idx", 0))
    return kept


def _get_prefill_chunk_size(kv_manager, session_id: str, device) -> int:
    """Return a prefill chunk size that preserves the CUDA block stride.

    The outer CUDA runners already round chunks to ``micro_block_size + 1``.
    The attention hook has its own internal chunk loop, though; leaving that
    loop at the raw 1024-token config split a 1028-token outer chunk into
    1024 + 4 and created the 252-token/3-token block pairs seen in validation.
    MLX has one contiguous dense tail, so its internal and external chunk
    boundaries never disagree.  Keep the same invariant here.
    """
    configured = os.environ.get("DKV_PREFILL_CHUNK_SIZE")
    if configured is not None:
        try:
            base = int(configured)
        except ValueError:
            base = 512
    else:
        cfg = getattr(kv_manager, "config", None)
        base = int(getattr(cfg, "prefill_chunk_size", 512))
    base = max(1, base)

    if getattr(device, "type", None) == "cuda" and hasattr(kv_manager, "get_session_micro_block_size"):
        try:
            capacity = max(2, int(kv_manager.get_session_micro_block_size(session_id)) + 1)
            base = ((base + capacity - 1) // capacity) * capacity
        except Exception:
            # Chunking must never make attention fail; the configured value is
            # still a valid fallback if the session is not initialized yet.
            pass
    return base



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Handles multiple cos/sin shapes emitted by different transformers versions:
      [seq, D]        — old transformers (pre-4.48)
      [B, seq, D]     — transformers 4.48+
      [B, 1, seq, D]  — already broadcast (no unsqueeze needed)
    The unsqueeze_dim arg is kept for backward compatibility but is now
    applied only when needed so the function is safe on all shapes.
    """
    # Normalise to [B, 1, seq, D] (broadcast over heads)
    if cos.dim() == 2:      # [seq, D] → old path
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    elif cos.dim() == 3:    # [B, seq, D] — transformers 5.x
        cos = cos.unsqueeze(1)   # → [B, 1, seq, D]
        sin = sin.unsqueeze(1)
    # dim==4: [B, 1, seq, D] — nothing to do

    # Partial RoPE (Qwen3.5/GLM-style: partial_rotary_factor < 1.0) — cos/sin's
    # last dim is only the rotary sub-range; rotate that slice and pass the
    # remainder through untouched, then concatenate back. When the model uses
    # full rotary (the common case), rotary_dim == q.shape[-1], q_pass is
    # empty, and this reduces to the original unconditional rotation exactly.
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _apply_rope_single(x, cos, sin):
    """Single-tensor counterpart of apply_rotary_pos_emb's partial-RoPE slicing,
    for the dense-history K reconstruction sites that only rotate K (Q is
    already rotated earlier via apply_rotary_pos_emb).
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


def _resolve_rotary_emb(model):
    """Walk common model layouts to find the rotary_emb module.

    Replaces hard 'model.model.rotary_emb' accesses so DKV works with
    non-standard model hierarchies (Falcon, GPT-NeoX, multimodal, etc.).
    Cached on the model object after the first call.
    """
    cached = getattr(model, "_dkv_resolved_rotary_emb", None)
    if cached is not None:
        return cached
    # Walk common attribute paths first (fast)
    for attr_path in (
        "model.rotary_emb",
        "rotary_emb",
        "transformer.rotary_emb",
        "model.model.rotary_emb",  # kept for legacy compat
    ):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if callable(obj):
                model._dkv_resolved_rotary_emb = obj
                return obj
        except AttributeError:
            continue
    # Slow path: scan all named modules
    for _, mod in model.named_modules():
        if "rotary" in type(mod).__name__.lower() and callable(mod):
            model._dkv_resolved_rotary_emb = mod
            return mod
    return None

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Universal repeat_kv implementation to support GQA.
    """
    if n_rep == 1:
        return hidden_states
    bs, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, slen, head_dim)
    out = hidden_states.reshape(bs, num_key_value_heads * n_rep, slen, head_dim)
    if hidden_states.device.type == "mps":
        # MPS scaled_dot_product_attention supports non-contiguous views natively
        return out
    return out.contiguous()

# ---------------------------------------------------------------------------
# PHASE 6: Fused Sparse Attention Integration
#
# Execution paths:
#   PREFILL (q_len > 1):  Dense path — required for causal masking over new tokens.
#   DECODE  (q_len == 1): FUSED SPARSE path — directly reads U/V/anchor without
#                          materializing a dense KV sequence via torch.cat().
#
# Performance notes (Phase 29):
#   - All module imports hoisted to top level — eliminates 84+ sys.modules lookups/token
#   - Ring buffer decode ingest — eliminates torch.cat allocations in hot path
#   - Metadata on CPU — eliminates 112+ CUDA sync points per token
#   - micro_block_size cached — eliminates O(N·L) block scan per token
# ---------------------------------------------------------------------------

def first_dkv_layer_index(layers) -> int:
    """Index of the first layer DKV actually attends — NOT model layer 0.

    Derived from the model at patch time, never hardcoded, so it is correct for
    any interleaving pattern rather than for the one hybrid that happened to be
    tested. Qwen3.5-2B puts attention at every 4th layer (first = 3); a model
    that put it at 0,1,2 then went linear gives 0; one that front-loads linear
    layers gives whatever index its first attention layer sits at. Nothing here
    knows the period, the count, or the model family — only "does this layer
    have self_attn", which is the same predicate the patch loop uses to decide
    what to wrap. The two therefore cannot disagree.

    Thirteen once-per-token gates in dkv_forward compare against this. They used
    to compare against 0, which is dead on any model whose layer 0 is not an
    attention layer -- see the note at the call site.
    """
    for i, l in enumerate(layers):
        if hasattr(l, "self_attn"):
            return i
    return 0        # no attention layers at all; the patch loop wraps nothing


def apply_dkv_attention_patch(model, kv_manager):
    """
    Monkey-patches the HF model's attention layers to route KV operations
    through our KVRuntimeManager.

    Gate: DKV_USE_ATTENTION_INTERFACE=1
        When set, delegates to the transformers 5.x AttentionInterface path
        (dkv_backend.register_dkv_backend + bind_kv_manager) instead of
        monkey-patching.  The model must have been loaded with
        attn_implementation='dkv' for this to take effect.

    Phase 6 change: decode step no longer calls kv_manager.get_kv() (which
    issues aten::cat over reconstructed blocks). Instead it calls
    kv_manager.get_raw_blocks() and passes them directly to
    fused_sparse_attention_decode().
    """
    # ── DKV_USE_ATTENTION_INTERFACE gate ──────────────────────────────────────
    # DEFAULT "0" (Path A, fused decode kernel) -- see the note at the matching
    # gate in serving/hf_dkv_wrapper.py for the measurement that motivated it.
    if os.environ.get("DKV_USE_ATTENTION_INTERFACE", "0") == "1":
        try:
            from runtime.dkv_backend import register_dkv_backend, bind_kv_manager
            register_dkv_backend(kv_manager=kv_manager, model_ref=model)
            bind_kv_manager(model, kv_manager)
            # Also stamp session_ids default (same as old patch)
            if not hasattr(model, "_dkv_session_ids"):
                model._dkv_session_ids = ["default"]
            print(
                "[DKV] DKV_USE_ATTENTION_INTERFACE=1: using AttentionInterface backend. "
                "Model must be loaded with attn_implementation='dkv'.",
                flush=True,
            )
            # Monkey-patch lm_head for last-token slicing (same as old path)
            if hasattr(model, "lm_head"):
                _orig_lm = model.lm_head.forward
                def _lm_head_sliced(hidden_states, _orig=_orig_lm):
                    if getattr(model, "_disable_lm_head_slicing", False):
                        return _orig(hidden_states)
                    if hidden_states.shape[1] > 1:
                        return _orig(hidden_states[:, -1:, :])
                    return _orig(hidden_states)
                model.lm_head.forward = _lm_head_sliced
            print(
                "DKV AttentionInterface Backend Active. [transformers 5.x path]",
                flush=True,
            )
            return
        except Exception as _e:
            print(
                f"[DKV] WARNING: AttentionInterface registration failed ({_e}). "
                "Falling back to legacy monkey-patch path.",
                flush=True,
            )
            import traceback
            traceback.print_exc()

    # ── Legacy monkey-patch path (default, DKV_USE_ATTENTION_INTERFACE=0) ─────
    # Some newer HF configs (e.g. Qwen3.5, other composite/multimodal models)
    # nest the text-decoder fields under `text_config` instead of exposing
    # them flat on `model.config` -- fall back to that before giving up.
    _cfg = model.config
    if not hasattr(_cfg, "num_attention_heads") and hasattr(_cfg, "text_config"):
        _cfg = _cfg.text_config
    num_heads             = _cfg.num_attention_heads
    num_key_value_heads   = getattr(_cfg, "num_key_value_heads", num_heads)
    hidden_size           = _cfg.hidden_size
    head_dim              = getattr(_cfg, "head_dim", None) or (hidden_size // num_heads)
    num_key_value_groups  = num_heads // num_key_value_heads

    # ── transformers-version convention detection ───────────────────────────
    # dkv_forward below must match whatever calling convention the installed
    # transformers version actually uses for THIS model's attention class.
    # 4.x callers pass hidden_states, attention_mask, position_ids,
    # past_key_value (singular), use_cache, output_attentions, cache_position,
    # position_embeddings -- all by keyword. 4.48+/5.x callers dropped
    # use_cache/output_attentions/cache_position as explicit kwargs (folded
    # into **kwargs) and renamed past_key_value -> past_key_values (plural).
    # Both conventions pass every argument BY KEYWORD (never positionally),
    # so dkv_forward accepts both names and normalizes internally; the one
    # thing that genuinely differs is how many values the caller unpacks the
    # return into (3-tuple for 4.x, 2-tuple for 5.x) -- detected once here.
    _new_cache_convention = False
    try:
        import inspect as _inspect
        # Hybrid architectures (Qwen3-Next/Qwen3.5-style) interleave attention-free
        # linear/gated-delta-net layers with no `self_attn` at all -- layer 0 may be
        # one of those, so find the first layer that actually has self_attn instead
        # of assuming index 0.
        _attn_layer = next(l for l in model.model.layers if hasattr(l, "self_attn"))
        _sig = _inspect.signature(type(_attn_layer.self_attn).forward)
        _params = set(_sig.parameters)
        _new_cache_convention = "past_key_values" in _params and "past_key_value" not in _params
        if not (({"attention_mask", "use_cache"} <= _params) or _new_cache_convention):
            import transformers as _tfm
            raise RuntimeError(
                "DKV CUDA attention interception does not recognize this "
                f"transformers {_tfm.__version__} attention forward signature "
                f"{tuple(_sig.parameters)} -- neither the 4.x nor the 4.48+/5.x "
                "convention this patch knows how to speak. This would silently "
                "produce garbage output, so refusing to patch instead."
            )
    except RuntimeError:
        raise
    except Exception:
        pass  # introspection failed — proceed and let the patch run

    # THE FIRST LAYER DKV ACTUALLY ATTENDS -- not model layer 0.
    #
    # Thirteen places in dkv_forward gate once-per-token work on
    # `captured_layer_idx == 0`: finalize_compressed_blocks (which PUBLISHES
    # background-compressed blocks to the pool), the DKV bypass decision, the
    # SRL router pre-warm, contiguous-prefill buffer cleanup, and more. Every
    # one of them assumes model layer 0 is a layer DKV sees.
    #
    # On a hybrid it is not. The loop below skips layers with no self_attn, so
    # on Qwen3.5-2B DKV attends 3, 7, 11, 15, 19, 23 and captured_layer_idx is
    # NEVER 0 -- confirmed by the route probe, which prints exactly those and no
    # layer 0. So on every hybrid model all thirteen gates are dead code.
    #
    # finalize_compressed_blocks being dead is the one with teeth: it is what
    # uploads a background-compressed block and writes it into the pool, i.e.
    # what moves a block from SUBMITTED to COMPRESSED. Without it a block is
    # compressed and then never published, which is precisely the stranded block
    # the coverage check keeps reporting:
    #     BLOCK COVERAGE: ... states=['SUBMITTED'] anchors=[1542]
    # and it explains why three separate producer-side fixes did not clear it --
    # the compression was fine, the publish step never ran.
    #
    # The block right above already applies this exact reasoning to find an
    # attention layer for signature introspection ("layer 0 may be one of those,
    # so find the first layer that actually has self_attn instead of assuming
    # index 0"). It just was not applied to the gates. Defining it once here
    # fixes all of them together and stays correct for any architecture, rather
    # than special-casing hybrids at thirteen call sites.
    _first_dkv_layer = first_dkv_layer_index(model.model.layers)

    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, "self_attn"):
            # Non-attention layer (linear/gated-delta-net) in a hybrid
            # architecture -- no KV cache here for DKV to intercept, leave
            # its native forward untouched.
            continue

        def make_dkv_forward(captured_layer_idx):
            def dkv_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value=None,
                past_key_values=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ):
                # Normalize the two cache-arg names (4.x: past_key_value singular,
                # 4.48+/5.x: past_key_values plural) into the one the rest of this
                # function already uses, and infer use_cache when the caller no
                # longer passes it explicitly (5.x signals caching purely via the
                # Cache object's presence).
                if past_key_value is None:
                    past_key_value = past_key_values
                if not use_cache and past_key_value is not None:
                    use_cache = True

                bsz, q_len, _ = hidden_states.size()

                # Zero-overhead bypass check
                is_decode = (use_cache and q_len == 1)
                is_bypassed = False
                session_ids = getattr(model, "_dkv_session_ids", ["default"] * bsz)
                sid = session_ids[0] if session_ids else None

                if not is_decode:
                    total_prompt_len = q_len
                    if sid and sid != "dummy_session":
                        if hasattr(kv_manager, "_session_token_ids"):
                            tokens = kv_manager._session_token_ids.get(sid)
                            if tokens is not None:
                                total_prompt_len = max(total_prompt_len, tokens.numel())
                        if hasattr(kv_manager, "get_session_sequence_length"):
                            total_prompt_len = max(total_prompt_len, kv_manager.get_session_sequence_length(sid))
                        if hasattr(kv_manager, "_streaming_mgr") and kv_manager._streaming_mgr is not None:
                            total_prompt_len = max(total_prompt_len, kv_manager._streaming_mgr.session_prefill_lens.get(sid, 0))
                    if total_prompt_len < _get_engage_threshold():
                        is_bypassed = True
                else:
                    # Decode step: bypass if we don't have any captured blocks in the manager.
                    # BUG (found 2026-07-27): this used to hardcode layer_idx=0, which
                    # silently assumes layer 0 is a normal attention layer with a KV
                    # cache. For hybrid architectures (Qwen3.5/Qwen3-Next-style, where
                    # most layers are linear_attention and only every Nth is
                    # full_attention) layer 0 is very often NOT one of the attended
                    # layers DKV ever compresses, so this check always saw zero blocks
                    # and forced every decode step onto the dense-only bypass path below
                    # -- silently ignoring the entire compressed KV pool, including any
                    # exact/force_exact-stored content. captured_layer_idx is THIS
                    # layer's own index, which is guaranteed to be an attended layer
                    # (only those get dkv_forward patched in), so it's always a correct
                    # probe -- unlike 0, which is only correct by coincidence on
                    # non-hybrid models where every layer is attended.
                    has_blocks = False
                    if sid and sid != "dummy_session":
                        if hasattr(kv_manager, "get_streaming_blocks"):
                            has_blocks = len(kv_manager.get_streaming_blocks(sid, captured_layer_idx)) > 0
                    if not has_blocks:
                        is_bypassed = True

                if is_bypassed:
                    cache_obj = None
                    if sid and sid != "dummy_session":
                        if hasattr(kv_manager, "decode_workspace"):
                            sess_dict = kv_manager.decode_workspace.setdefault(sid, {})
                            if "dense_cache" not in sess_dict:
                                from transformers.cache_utils import DynamicCache
                                sess_dict["dense_cache"] = DynamicCache()
                            cache_obj = sess_dict["dense_cache"]

                    kwargs_clean = kwargs.copy()
                    # Pass ONLY the cache kwarg this transformers version's
                    # attention actually accepts. 4.46 uses `past_key_value`
                    # (singular); passing the plural too → "unexpected keyword
                    # argument 'past_key_values'". Introspect once (cached).
                    _op_params = getattr(self, "_dkv_orig_params", None)
                    if _op_params is None:
                        import inspect as _insp
                        try:
                            _op_params = set(_insp.signature(self._original_forward).parameters)
                        except (ValueError, TypeError):
                            _op_params = {"past_key_value"}
                        self._dkv_orig_params = _op_params
                    if "past_key_value" in _op_params:
                        kwargs_clean["past_key_value"] = cache_obj
                    if "past_key_values" in _op_params:
                        kwargs_clean["past_key_values"] = cache_obj

                    # Multi-chunk bypass correctness. The wrapper prefills in
                    # chunks and calls the model WITHOUT past_key_values, so the
                    # model sees key_value_length == query_length, takes the SDPA
                    # "ignore mask" fast path and hands us attention_mask=None.
                    # But we thread our own dense_cache here, so key_states holds
                    # (prefix_len + q_len) keys while there are only q_len queries.
                    # F.scaled_dot_product_attention(is_causal=True) on that
                    # non-square shape uses UPPER-LEFT alignment (query i attends
                    # keys 0..i) instead of the correct lower-right window
                    # (query i attends keys 0..prefix_len+i) — so chunk 2+ queries
                    # attend the wrong keys and prefill logits are garbage.
                    # Square chunk 1 / single-forward prefill are unaffected.
                    # Fix: build the explicit lower-right causal mask whenever a
                    # cached prefix exists. Per-layer prefix length is read BEFORE
                    # this layer's cache update (get_seq_length respects layer_idx).
                    if (cache_obj is not None and attention_mask is None
                            and q_len > 1):
                        _lidx = getattr(self, "layer_idx", 0) or 0
                        try:
                            _prefix_len = cache_obj.get_seq_length(_lidx)
                        except Exception:
                            _prefix_len = 0
                        if _prefix_len > 0:
                            _kv_len = _prefix_len + q_len
                            _dev = hidden_states.device
                            _dt = hidden_states.dtype
                            _row = torch.arange(q_len, device=_dev).unsqueeze(1)      # [q,1]
                            _col = torch.arange(_kv_len, device=_dev).unsqueeze(0)    # [1,kv]
                            _allowed = _col <= (_prefix_len + _row)                   # [q,kv] bool
                            _m = torch.zeros((q_len, _kv_len), dtype=_dt, device=_dev)
                            _m.masked_fill_(~_allowed, torch.finfo(_dt).min)
                            attention_mask = _m.view(1, 1, q_len, _kv_len)

                    return self._original_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs_clean
                    )





                # Helpers for Decomposed Prefill Attention (Part 1)
                def _flash_local_attention(q, k, v):
                    # Local causal self-attention over the new chunk.
                    # Compute scores once and derive both the output AND the logsumexp
                    # from them — avoids the previous double-computation where both
                    # F.scaled_dot_product_attention AND torch.matmul were called,
                    # which doubled peak memory per chunk/layer.
                    q_len_inner = q.shape[2]
                    k_rep = repeat_kv(k, num_key_value_groups)
                    v_rep = repeat_kv(v, num_key_value_groups)

                    scale = 1.0 / math.sqrt(head_dim)
                    scores = torch.matmul(q * scale, k_rep.transpose(-2, -1))

                    # Causal mask — use q.dtype to stay in fp16 on MPS
                    mask = torch.triu(
                        torch.full((q_len_inner, q_len_inner), float('-inf'),
                                   device=q.device, dtype=q.dtype), diagonal=1
                    )
                    scores = scores + mask

                    lse = torch.logsumexp(scores, dim=-1)      # [B, H, q_len]
                    weights = torch.softmax(scores, dim=-1)    # [B, H, q_len, k_len]
                    out = torch.matmul(weights, v_rep)         # [B, H, q_len, D]
                    del scores, weights, mask
                    return out, lse

                def _project_then_attend_history(q, comp_blocks, pool, sid, cos_all=None, sin_all=None):
                    # O5b: Full history attention path routed through the JIT-fused kernel.
                    # All 8 inner ops (K-recon, cat, RoPE, score, mask, logsumexp, softmax,
                    # value-reduction) execute in a single TorchScript compilation unit.
                    _, H, q_len_inner, head_dim = q.shape
                    pool_indices = [b.pool_idx for b in comp_blocks]
                    pool_indices_t = torch.tensor(pool_indices, device=q.device, dtype=torch.long)

                    N_blocks  = len(comp_blocks)
                    max_seq_len = pool.U.shape[1]

                    # Gather block data from the pool
                    U_stack    = reconstruct_batch_U(pool, pool_indices_t).to(q.dtype)
                    V_K_stack  = pool.V_K[pool_indices_t].to(q.dtype)        # [N, R, num_kv_heads, D]
                    V_V_stack  = pool.V_V[pool_indices_t].to(q.dtype)        # [N, R, num_kv_heads, D]
                    anc_K      = torch.stack([b.anchor_kv[0, 0] for b in comp_blocks], dim=0).to(q.dtype)  # [N, num_kv_heads, D]
                    anc_V      = torch.stack([b.anchor_kv[0, 1] for b in comp_blocks], dim=0).to(q.dtype)  # [N, num_kv_heads, D]
                    scales_1d  = pool.scales[pool_indices_t]     # [N]
                    seq_lens_t = pool.seq_lens[pool_indices_t]   # [N] int32

                    # Gather residuals
                    res_K_pos = pool.residual_K_positions[pool_indices_t]
                    res_K_val_raw = pool.residual_K_values[pool_indices_t].to(q.dtype)
                    res_K_val = repeat_kv(res_K_val_raw.permute(0, 2, 1, 3), num_key_value_groups).permute(0, 2, 1, 3)
                    res_V_pos = pool.residual_V_positions[pool_indices_t]
                    res_V_val_raw = pool.residual_V_values[pool_indices_t].to(q.dtype)
                    res_V_val = repeat_kv(res_V_val_raw.permute(0, 2, 1, 3), num_key_value_groups).permute(0, 2, 1, 3)

                    # GQA repeat-expansion (zero-copy expand → contiguous only if needed)
                    v_k_rep   = repeat_kv(V_K_stack.permute(0, 2, 1, 3), num_key_value_groups)   # [N, H, R, D]
                    v_v_rep   = repeat_kv(V_V_stack.permute(0, 2, 1, 3), num_key_value_groups)   # [N, H, R, D]
                    anc_k_rep = repeat_kv(anc_K.unsqueeze(2), num_key_value_groups).squeeze(2)   # [N, H, D]
                    anc_v_rep = repeat_kv(anc_V.unsqueeze(2), num_key_value_groups).squeeze(2)   # [N, H, D]

                    # Cache-aware RoPE slicing to eliminate redundant GPU gathers per layer
                    anchors_tuple = tuple(b.anchor_idx for b in comp_blocks)
                    session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                    prefill_cos_cache = session_dict.setdefault("prefill_cos_sliced", {})
                    prefill_sin_cache = session_dict.setdefault("prefill_sin_sliced", {})
                    cos_sliced = prefill_cos_cache.get(anchors_tuple)
                    sin_sliced = prefill_sin_cache.get(anchors_tuple)

                    if cos_sliced is None or sin_sliced is None:
                        block_anchors   = torch.tensor(anchors_tuple, device=q.device, dtype=torch.long)
                        positions       = block_anchors.view(N_blocks, 1) + torch.arange(1 + max_seq_len, device=q.device).view(1, 1 + max_seq_len)
                        positions_flat  = positions.reshape(-1)
                        cos_ref = cos_all if cos_all is not None else cos
                        sin_ref = sin_all if sin_all is not None else sin
                        # Clamp positions to actual sequence length of cos_ref to prevent indexing out of bounds
                        seq_len_limit = cos_ref.shape[1] if cos_ref.dim() >= 3 else cos_ref.shape[0]
                        positions_flat = positions_flat.clamp(min=0, max=seq_len_limit - 1).clone()
                        cos_gathered = cos_ref[0, positions_flat]
                        sin_gathered = sin_ref[0, positions_flat]
                        rotary_dim = cos_gathered.shape[-1]
                        if rotary_dim < head_dim:
                            # Partial RoPE (Qwen3.5-style) -- same pad-to-head_dim rationale
                            # as the decode-side anchor cos/sin construction above; see that
                            # comment for the full explanation and its known approximation.
                            _pad_shape = cos_gathered.shape[:-1] + (head_dim - rotary_dim,)
                            cos_gathered = torch.cat(
                                [cos_gathered, torch.ones(_pad_shape, device=cos_gathered.device, dtype=cos_gathered.dtype)], dim=-1)
                            sin_gathered = torch.cat(
                                [sin_gathered, torch.zeros(_pad_shape, device=sin_gathered.device, dtype=sin_gathered.dtype)], dim=-1)
                        cos_sliced = cos_gathered.view(N_blocks, 1 + max_seq_len, 1, head_dim)
                        sin_sliced = sin_gathered.view(N_blocks, 1 + max_seq_len, 1, head_dim)
                        prefill_cos_cache[anchors_tuple] = cos_sliced
                        prefill_sin_cache[anchors_tuple] = sin_sliced

                        # Evict stale prefill sliced RoPE cache entries in O(1) without scanning other keys
                        stale_keys = [k for k in list(prefill_cos_cache.keys()) if k != anchors_tuple]
                        for k in stale_keys:
                            prefill_cos_cache.pop(k, None)
                            prefill_sin_cache.pop(k, None)

                    inv_scale_val = 1.0 / math.sqrt(head_dim)

                    # ── O5b: Single JIT dispatch covering all inner math ─────────────
                    # result[0] = out_hist  [1, H, q_len, D]
                    # result[1, 0, :, :, 0] = lse_hist [H, q_len]  (last dim replicated)
                    result = _prefill_fused_history_attend(
                        U          = U_stack,
                        V_K        = v_k_rep.permute(0, 2, 1, 3),   # [N, R, H, D]
                        V_V        = v_v_rep.permute(0, 2, 1, 3),   # [N, R, H, D]
                        anchors_K  = anc_k_rep,
                        anchors_V  = anc_v_rep,
                        scales     = scales_1d,
                        cos_sliced = cos_sliced,
                        sin_sliced = sin_sliced,
                        q          = q,
                        seq_lens   = seq_lens_t,
                        inv_scale  = inv_scale_val,
                        residual_K_positions = res_K_pos,
                        residual_K_values    = res_K_val,
                        residual_V_positions = res_V_pos,
                        residual_V_values    = res_V_val,
                        # Resolved HERE, not inside the callee: that function is
                        # torch.jit.script'ed and TorchScript cannot read os.environ.
                        exact_residual = _exact_residual_semantics(q.device),
                    )
                    out_hist  = result[0]                     # [1, H, q_len, D]
                    lse_hist  = result[1, 0, :, :, 0]        # [H, q_len]
                    lse_hist  = lse_hist.unsqueeze(0)        # [1, H, q_len]  — matches _combine_outputs API
                    return out_hist, lse_hist

                def _combine_outputs(out_lse_list):
                    # Filter out empty paths
                    valid_list = [x for x in out_lse_list if x[0] is not None and x[1] is not None]
                    if not valid_list:
                        return None
                    if len(valid_list) == 1:
                        return valid_list[0][0]
                    
                    # Log-sum-exp stable online softmax combination
                    lses = torch.stack([x[1] for x in valid_list], dim=0)  # [M, 1, H, q_len]
                    lse_max, _ = torch.max(lses, dim=0)                  # [1, H, q_len]
                    
                    weights_list = []
                    denom = torch.zeros_like(lse_max)
                    for out, lse in valid_list:
                        w = torch.exp(lse - lse_max)
                        weights_list.append(w)
                        denom = denom + w
                    
                    # Prevent divide-by-zero
                    denom = torch.clamp(denom, min=1e-9)
                    
                    out_combined = torch.zeros_like(valid_list[0][0])
                    for idx, (out, lse) in enumerate(valid_list):
                        w = weights_list[idx] / denom
                        out_combined = out_combined + out * w.unsqueeze(-1)
                    
                    # Ensure dtype is cast back to avoid precision mismatch during linear projection
                    return out_combined.to(valid_list[0][0].dtype)

                # --- Projection ---
                # Support both split (q/k/v_proj) and fused (qkv_proj) layouts.
                _has_fused_qkv = (
                    hasattr(self, "qkv_proj")
                    and not (hasattr(self, "q_proj") and hasattr(self, "k_proj"))
                )
                if _has_fused_qkv:
                    _qkv = self.qkv_proj(hidden_states)
                    _q_sz = num_heads * head_dim
                    _k_sz = num_key_value_heads * head_dim
                    query_states = _qkv[..., :_q_sz]
                    key_states   = _qkv[..., _q_sz:_q_sz + _k_sz]
                    value_states = _qkv[..., _q_sz + _k_sz:]
                else:
                    query_states = self.q_proj(hidden_states)
                    key_states   = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                # Gated-attention variants (Qwen3-Next/Qwen3.5-style) pack [query | gate]
                # per head into q_proj's output (2x width): reshape to (..., n_heads,
                # 2*head_dim) and chunk the LAST axis so each head's own gate stays
                # paired with that head's query (a flat chunk(2) on the un-reshaped
                # tensor would wrongly split heads 0-3 from heads 4-7 instead).
                # sigmoid(attn_gate) is applied to the attention output right before
                # o_proj at each of the three return sites below, mirroring the
                # model's own attention class and MLX's attention_forward.
                attn_gate = None
                if query_states.shape[-1] == num_heads * head_dim * 2:
                    query_states = query_states.view(bsz, q_len, num_heads, head_dim * 2)
                    query_states, attn_gate = query_states.chunk(2, dim=-1)
                    attn_gate = attn_gate.reshape(bsz, q_len, -1)
                    query_states = query_states.transpose(1, 2)
                else:
                    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states   = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

                # --- QK-norm (Qwen3 / Gemma3 / models with per-head normalization) ---
                # CORRECTNESS FIX: skipping q_norm/k_norm produces wrong attention scores.
                # Gate: always applied when the attribute exists (no env flag needed —
                # not applying it is always wrong).
                if hasattr(self, "q_norm"):
                    query_states = self.q_norm(query_states)
                if hasattr(self, "k_norm"):
                    key_states = self.k_norm(key_states)

                # --- RoPE ---
                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                unrot_key_states = key_states.clone()
                # unrot_query_states holds the PRE-RoPE query.  In DECODE (q_len==1)
                # the block router reads it (the q_for_routing / raw_q uses, both
                # inside the is_decode branch).  In PREFILL (q_len>1) the ONLY use
                # is the last token, for SRL router pre-warm just below.  Cloning
                # the full [B, H, q_len, D] query every layer every prefill chunk
                # was pure waste (~10 MB/layer at a 1k chunk, ×48 layers); keep
                # only the last token in prefill.
                if q_len == 1:
                    unrot_query_states = query_states.clone()
                else:
                    unrot_query_states = query_states[:, :, -1:, :].clone()
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                session_ids = getattr(model, "_dkv_session_ids", ["default"] * bsz)

                # Fix 3: Gate finalize_compressed_blocks to layer 0 only.
                # The function is idempotent and protected by _pending_lock internally.
                # Calling it on all 28 layers = 27 wasted lock acquisitions per token.
                # At 1024-token generation that is 27x1024 = 27,648 unnecessary mutex ops —
                # the root cause of the -49% TPS regression at 1024 tokens.
                if use_cache and captured_layer_idx == _first_dkv_layer and hasattr(kv_manager, "finalize_compressed_blocks"):
                    kv_manager.finalize_compressed_blocks()

                # Track last prefill token query for SRL router pre-warming
                if captured_layer_idx == _first_dkv_layer and q_len > 1:
                    if not hasattr(kv_manager, "_last_prefill_q"):
                        kv_manager._last_prefill_q = {}
                    for b_idx, sid in enumerate(session_ids):
                        if sid != "dummy_session":
                            kv_manager._last_prefill_q[sid] = unrot_query_states[b_idx, :, -1, :].clone().detach()

                # ==================================================================
                # PHASE 6 BRANCHING
                # ==================================================================
                is_decode = (use_cache and q_len == 1)

                # Fix 2: Context-aware DKV bypass.
                # For prefill sessions below DKV_ENGAGE_THRESHOLD tokens with no
                # compressed history, all DKV overhead is pure cost with zero benefit
                # — there is nothing to compress and nothing to retrieve from the pool.
                # Route directly to standard SDPA: identical output to Dense baseline.
                # Decode is never bypassed: by decode time the session either has
                # compressed history (long context) or naturally falls through to the
                # has_dense-only SDPA branch (short context, no compressed blocks).
                if (not is_decode                           # prefill path only
                        and use_cache                       # single-request serving path
                        and captured_layer_idx == _first_dkv_layer):       # compute check once at layer 0
                    _engage_threshold = _get_engage_threshold()
                    _total_ctx = q_len
                    _primary_sid = session_ids[0] if session_ids else None
                    if _primary_sid and _primary_sid != "dummy_session":
                        if hasattr(kv_manager, "_session_token_ids"):
                            tokens = kv_manager._session_token_ids.get(_primary_sid)
                            if tokens is not None:
                                _total_ctx = max(_total_ctx, tokens.numel())
                        if hasattr(kv_manager, "get_session_sequence_length"):
                            _total_ctx = max(_total_ctx, kv_manager.get_session_sequence_length(_primary_sid))
                        if hasattr(kv_manager, "_streaming_mgr") and kv_manager._streaming_mgr is not None:
                            _total_ctx = max(_total_ctx, kv_manager._streaming_mgr.session_prefill_lens.get(_primary_sid, 0))
                    _has_history = False
                    if _total_ctx >= _engage_threshold:
                        _has_history = True  # don't bypass — context is long enough
                    else:
                        # Same hardcoded-layer-0 hazard as the decode-side check below
                        # (see its comment) -- captured_layer_idx is this closure's own
                        # attended layer, always a valid probe on hybrid models.
                        for _sid in session_ids:
                            if _sid != "dummy_session":
                                if kv_manager.get_streaming_blocks(_sid, captured_layer_idx):
                                    _has_history = True
                                    break
                    # Store bypass decision on the kv_manager for reuse by layers 1-27
                    kv_manager._bypass_this_step = (not _has_history)

                if (not is_decode
                        and getattr(kv_manager, "_bypass_this_step", False)):
                    # Pure Dense path — all 28 layers use this shortcut
                    k_rep = repeat_kv(key_states, num_key_value_groups)
                    v_rep = repeat_kv(value_states, num_key_value_groups)
                    attn_out = F.scaled_dot_product_attention(
                        query_states, k_rep, v_rep, is_causal=(q_len > 1)
                    )
                    attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_size)
                    if attn_gate is not None:
                        attn_out = attn_out * torch.sigmoid(attn_gate)
                    attn_out = self.o_proj(attn_out)

                    # Capture KV states so they are stored in the KV manager/pool for decode
                    for b_idx, sid in enumerate(session_ids):
                        if sid != "dummy_session":
                            kv_manager.capture_prefill_kv(
                                sid, captured_layer_idx,
                                _ingest_k(key_states, unrot_key_states)[b_idx:b_idx+1].detach(),
                                value_states[b_idx:b_idx+1].detach(),
                            )

                    # transformers 4.44-4.47 decoder unpacks a FIXED 3-tuple
                    # (attn_output, attn_weights, present_key_value). DKV keeps
                    # KV in the manager, so we return the cache object 4.46 passed
                    # in (past_key_value) UNCHANGED — the model finalizes it
                    # (to_legacy_cache); returning None → "NoneType has no
                    # to_legacy_cache". The old conditional tuple returned only 2.
                    if _new_cache_convention:
                        return (attn_out, None)
                    return (attn_out, None, past_key_value)

                if is_decode:
                    # ----------------------------------------------------------
                    # DECODE PATH — sparse execution via Triton or PyTorch fallback
                    # ----------------------------------------------------------

                    # Free the contiguous-prefill rotated buffer (DKV_CONTIGUOUS_PREFILL)
                    # now that prefill is over — it is only used by the prefill forward.
                    # Layer 0 clears every session's buffers; the compressed pool is the
                    # decode-time store from here on.
                    if captured_layer_idx == _first_dkv_layer and getattr(kv_manager, "_contig_prefill", None):
                        for _sid in session_ids:
                            kv_manager._contig_prefill.pop(_sid, None)

                    # 1. Ingest new tokens for all active batch elements
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        if sid == "dummy_session":
                            continue
                        # DKV_ROTATED_POOL: ingest POST-RoPE keys, which is what
                        # MLX does (mlx_dkv_wrapper.py:4613 passes keys_rot).
                        # key_states was rotated in place at :900; unrot_key_states
                        # is the pre-RoPE clone taken at :888. See
                        # triton_fused_decode.pool_stores_rotated_k for why the
                        # unrotated convention costs every compressed token a RoPE
                        # phase error of up to a full block.
                        curr_k = (key_states[b_idx:b_idx+1] if _pool_rotated_k()
                                  else unrot_key_states[b_idx:b_idx+1])
                        curr_v = value_states[b_idx:b_idx+1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)
                        if captured_layer_idx == _first_dkv_layer:
                            srl_state = kv_manager.get_srl_state(sid)
                            if srl_state is not None:
                                # Accumulate every 8 tokens to amortize the D2H copy cost.
                                # recent_decode_keys is only used for SRL re-routing heuristics,
                                # so coarse sampling is fine.
                                # CUDA-graph stage 1: this .cpu() is the one host sync on the
                                # SRL decode path (the base low/mid/high presets never enter
                                # here — srl_state is None).  DKV_GRAPH_SAFE_DECODE=1 skips
                                # it so the whole decode forward is provably sync-free and can
                                # be captured; the only cost is the SRL re-routing heuristic
                                # loses its recent-key trail (routing still works from anchors).
                                _step_ctr = getattr(srl_state, "_decode_step_ctr", 0)
                                srl_state._decode_step_ctr = _step_ctr + 1
                                if _step_ctr % 8 == 0 and os.environ.get("DKV_GRAPH_SAFE_DECODE", "0") != "1":
                                    k_avg = curr_k[0].mean(dim=0).squeeze(0).float().cpu() # [head_dim]
                                    srl_state.recent_decode_keys.append(k_avg)
                                    if len(srl_state.recent_decode_keys) > 512:
                                        srl_state.recent_decode_keys = srl_state.recent_decode_keys[-512:]

                    # 2. Attention decode dispatch
                    # Always route through the unified native_triton_sparse_attn_decode path.
                    # This dispatches to either the GPU Triton kernel or the high-performance
                    # Project-Then-Attend PyTorch fallback!
                    attn_outputs = []
                    # P1-6: Deferred Triton batch dispatch for CUDA combined path.
                    # When bsz > 1, collecting all session params then dispatching in tight
                    # sequence eliminates per-session Python overhead between kernel launches,
                    # allowing CUDA to pipeline them. Each call is still B=1 (no kernel change).
                    _triton_batch_queue = []   # list of (b_idx, kwargs) for deferred dispatch
                    _triton_batch_enabled = (
                        bsz > 1
                        and HAS_TRITON
                        and query_states.device.type == "cuda"
                        and os.environ.get("DKV_BATCH_TRITON_DISPATCH", "1") not in ("0", "false", "off")
                    )
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        if sid == "dummy_session":
                            attn_outputs.append(
                                torch.zeros(
                                    (1, num_heads, 1, head_dim),
                                    device=query_states.device, dtype=query_states.dtype
                                )
                            )
                            continue

                        block_indices, dense_blocks, anchor_indices, max_anchor_idx, max_valid_len = kv_manager.get_cached_decode_blocks(
                            sid, captured_layer_idx, query_states.device
                        )
                        pool = getattr(kv_manager, 'native_pool', None)
                        session_mbs = kv_manager.get_session_micro_block_size(sid)

                        # ── SRL Routing ────────────────────────────────────────────
                        # On layer 0: compute selected_slots and cache on srl_state.
                        # On layers 1-27: reuse the cached slots (cost paid once).
                        # Bypass if: SRL not built, pool not ready, or too few blocks.
                        srl_state = None
                        _srl_rerouted = False
                        srl_enabled = True
                        if pool is not None and pool.W_proj is not None:
                            srl_state = kv_manager.get_srl_state(sid)
                            if srl_state is not None:
                                if captured_layer_idx == _first_dkv_layer:
                                    srl_state.current_step_factual_tokens = set()
                                    srl_state.current_step_factual_sequences = []
                                    srl_state.current_step_max_similarity = 0.0
                                    srl_state.current_step_sequence_entity_ids = []
                                    srl_state.current_step_sequence_is_prime = []
                                    srl_state.current_step_sequence_prefixes = []
                                    # Invalidate per-step caches
                                    srl_state._prime_tokens_cache = None
                                session_config = getattr(kv_manager, "session_configs", {}).get(sid, {})
                                srl_enabled = session_config.get("srl_enabled", True)

                        # ── How many blocks before routing engages ───────────
                        # MLX has ONE condition:  if topk_blocks > 0 and nb > k_eff
                        # (mlx_dkv_wrapper.py:4021) -- routing engages as soon as
                        # there are more blocks than the budget, i.e. from 17.
                        #
                        # This side ALSO required block_indices.numel() >
                        # srl_state.routing_threshold, a CUDA-only gate with no
                        # MLX counterpart. routing_threshold comes from the model
                        # size (hf_dkv_wrapper.py:704-710): 25 / 40 / 50 by param
                        # count, so Qwen3.5-2B gets 40. With a 257-token block and
                        # the recency window excluded that lands as:
                        #
                        #    2k  (2822 tok)   ~6 compressed blocks   6 > 40  NO
                        #    8k  (10903 tok) ~38 compressed blocks  38 > 40  NO
                        #    32k (32571 tok) ~122 compressed blocks 122 > 40 YES
                        #
                        # Two consequences, both observed:
                        #
                        #  1. Routing never ran at 2k or 8k, so every routing fix
                        #     produced byte-identical results there -- the same
                        #     "my change had no effect" signature that has already
                        #     cost this investigation several rounds.
                        #
                        #  2. 8k sits ON the boundary. Blocks keep compressing as
                        #     decode proceeds, so the count crosses 40 PART WAY
                        #     THROUGH a generation, and which token it crosses at
                        #     depends on when async compression lands. Routing
                        #     switching on mid-generation, at a nondeterministic
                        #     moment, is a temperature-0 run that does not
                        #     reproduce -- which is exactly what 8k@depth0.5 shows
                        #     (2/3 recall, 2 distinct outputs) while 8k@0.0 and
                        #     8k@0.9 are stable.
                        #
                        # The gate is also REDUNDANT for the default router:
                        # route_blocks_relevance already returns block_indices
                        # unchanged when N <= k_eff. So all the threshold adds is
                        # a SECOND, different boundary that no reference has.
                        # Keep it only for the legacy "srl" router, which was
                        # tuned with it.
                        _router_mode_gate = os.environ.get("DKV_ROUTER", "residual").lower()
                        _route_min = (srl_state.routing_threshold
                                      if (_router_mode_gate == "srl" and srl_state is not None)
                                      else 0)
                        # ── ROUTING REACHABILITY PROBE ───────────────────────
                        # Say ONCE per session/layer whether the router ran, and
                        # if not, WHICH condition stopped it.
                        #
                        # This exists because "the change had no effect" has now
                        # been the wrong conclusion three times, from three
                        # different mechanisms: a swallowed exception (1b0de37),
                        # an unreachable branch (c3514e0) and an outer gate
                        # (140748b). Each time the algorithm was edited before
                        # anyone checked the code ran, and each time the edit was
                        # inert. The BLOCK COVERAGE check ended that guessing for
                        # block reachability by simply reporting the fact; this
                        # is the same instrument for routing.
                        #
                        # Costs one dict lookup per layer after the first token.
                        if os.environ.get("DKV_ROUTE_PROBE", "1") != "0":
                            _rp = getattr(kv_manager, "_route_probe_seen", None)
                            if _rp is None:
                                _rp = kv_manager._route_probe_seen = set()
                            # Keyed on the CANDIDATE COUNT too, not just
                            # (session, layer). Every prompt in a harness run
                            # reuses session "default", so a (sid, layer) key
                            # printed for the first prompt and stayed silent for
                            # every one after it -- the 2k case reported while 8k
                            # and 32k, the ones actually in question, did not.
                            _rp_key = (sid, captured_layer_idx,
                                       block_indices.numel() if block_indices is not None else -1)
                            if _rp_key not in _rp:
                                _rp.add(_rp_key)
                                _n_blocks = (block_indices.numel()
                                             if block_indices is not None else 0)
                                if not srl_enabled:
                                    _why = "srl_enabled=False (session config)"
                                elif srl_state is None:
                                    _why = ("srl_state is None -- SRL index not built "
                                            "(pool.W_proj missing, or finalize_srl_index "
                                            "never ran for this session)")
                                elif block_indices is None:
                                    _why = "block_indices is None -- no compressed blocks"
                                elif _n_blocks <= _route_min:
                                    _why = (f"only {_n_blocks} candidate block(s), "
                                            f"needs > {_route_min}")
                                else:
                                    _why = None
                                print(f"[DKV] ROUTE PROBE layer={captured_layer_idx} "
                                      f"session={sid} candidates={_n_blocks} "
                                      f"router={_router_mode_gate} "
                                      + ("ROUTING RUNS" if _why is None
                                         else f"ROUTER SKIPPED: {_why}"), flush=True)

                        if (
                            srl_enabled
                            and srl_state is not None
                            and block_indices is not None
                            and block_indices.numel() > _route_min
                        ):
                            try:
                                from native_core.srl.query_router import route_query_fixed_k
                                # DKV_ROUTE_PER_LAYER=1: opt out of cross-layer route sharing
                                # (MLX parity -- MLX's default is to route every layer
                                # independently; DKV_ROUTE_ONCE=1 is MLX's opt-IN to sharing,
                                # for the opposite tradeoff). CUDA's default here is the
                                # reverse -- share the layer-0 decision across all layers
                                # unconditionally -- kept as the default for backward
                                # compatibility (existing perf characteristics), with this
                                # as an accuracy-over-speed escape hatch.
                                # DEFAULT FLIPPED TO PER-LAYER: MLX PARITY, AND
                                # THE LAYER-0 ASSUMPTION IS BROKEN ON HYBRIDS.
                                #
                                # The comment above already records the parity
                                # gap -- MLX routes every layer independently and
                                # treats SHARING as the opt-in. But sharing is
                                # worse than a policy difference here, because
                                # the sharing branch keys on
                                #
                                #     captured_layer_idx == 0
                                #
                                # and captured_layer_idx is the MODEL's layer
                                # index. Qwen3.5-2B is a hybrid: only every 4th
                                # layer is full-attention, so DKV sees layers
                                # 3, 7, 11, 15, 19, 23 and NEVER 0. The route
                                # probe confirms it -- it prints layer=3 .. 23,
                                # no layer 0. So the selection branch never runs,
                                # every layer falls to "reuse the layer-0
                                # decision", and there is no layer-0 decision to
                                # reuse: current_step_slots stays None and the
                                # reroute is skipped entirely.
                                #
                                # This file already flags the same trap elsewhere
                                # ("the same hardcoded-layer-0 hazard as the
                                # decode-side check below -- captured_layer_idx
                                # is this closure's own attended layer, always a
                                # valid probe on hybrid models"), so it was known
                                # for one check and left standing for this one.
                                #
                                # Routing per layer removes the assumption rather
                                # than working around it, and is what the
                                # reference does. Cost is one relevance BMM per
                                # DKV layer per routing interval -- six on this
                                # model, over an NxD anchor matrix -- not per
                                # token. DKV_ROUTE_ONCE=1 restores sharing (which
                                # is MLX's own name for the opt-in).
                                _route_per_layer = (
                                    os.environ.get("DKV_ROUTE_PER_LAYER", "1") == "1"
                                    and os.environ.get("DKV_ROUTE_ONCE", "0") != "1")
                                if captured_layer_idx == _first_dkv_layer or _route_per_layer:
                                    # SRL routing cadence: route every N tokens to amortise
                                    # the D2H cost of entropy/.item(), centroid/.tolist(),
                                    # and semantic score vector .cpu() in route_query_fixed_k.
                                    # Between route steps, cached slots from the previous step
                                    # are reused (valid because token embeddings change slowly).
                                    # Default N=1 = every token (original behaviour).
                                    # DKV_SRL_ROUTE_EVERY=4 routes every 4 tokens (~3-4×
                                    # less D2H traffic during long decodes).
                                    # MLX PARITY — ROUTE ON AN INTERVAL AND HOLD.
                                    #
                                    # MLX routes once per DKV_DECODE_CACHE_INTERVAL (16) tokens
                                    # and holds that selection for the whole answer:
                                    #     need_route = (ent is None)
                                    #                  or (ent["steps"] >= self._decode_cache_interval)
                                    #                  or (ent.get("nb") != nb)
                                    #                                 (mlx_dkv_wrapper.py:4002)
                                    # This side re-routed EVERY TOKEN, and the route trace shows
                                    # exactly what that costs. At 32k@depth0.9 the needle's block
                                    # is ranked 0-1 of 122 at the FIRST decode token (res_max
                                    # 11.5-15.0, matching MLX's rank 0 / 15.1-19.1) and then
                                    # collapses to rank 13-111 / res_max 2-7 on later tokens --
                                    # because the query by then is '<think>', '\n\n', '</think>',
                                    # i.e. content-free, and a content-free query cannot rank the
                                    # block holding a buried code. The answer is emitted around
                                    # step 5, so the routing that mattered had already been
                                    # thrown away and re-derived from a token that knows nothing.
                                    #
                                    # It also explains the partial codes. Recall tracks exactly
                                    # how long good routing survives: 32k@0.0 and @0.5 hold it
                                    # ~10 tokens -> 2/3, and their failures are 'ZEBRA-447' /
                                    # 'ZEBRA-4471' -- the code starts right and degrades mid-
                                    # emission as the block drops out. 32k@0.9 holds it ONE token
                                    # -> 0/3 and 'None'.
                                    #
                                    # THE COUNTER NEVER ADVANCED, so this gate was dead and the
                                    # cadence knob did nothing: current_step_count is incremented
                                    # only in query_router.route_query (:894), the LEGACY srl
                                    # router, which the default DKV_ROUTER=residual never calls.
                                    # `_should_route` was therefore `0 % N == 0` -> always True.
                                    # It is advanced here instead, once per TOKEN at the first
                                    # DKV layer (not once per layer, which would multiply the
                                    # rate by the layer count and re-route every token again).
                                    _route_every = getattr(srl_state, "_route_cadence", None)
                                    if _route_every is None:
                                        # Default to MLX's own interval, read from MLX's own
                                        # variable so the two cannot drift; DKV_SRL_ROUTE_EVERY
                                        # still overrides for A/B.
                                        try:
                                            _mlx_iv = int(os.environ.get("DKV_DECODE_CACHE_INTERVAL", "16"))
                                        except (ValueError, TypeError):
                                            _mlx_iv = 16
                                        try:
                                            _route_every = int(os.environ.get("DKV_SRL_ROUTE_EVERY", str(_mlx_iv)))
                                        except (ValueError, TypeError):
                                            _route_every = _mlx_iv
                                        srl_state._route_cadence = max(1, _route_every)

                                    if captured_layer_idx == _first_dkv_layer:
                                        srl_state.current_step_count = getattr(
                                            srl_state, "current_step_count", 0) + 1
                                    _step_ctr = getattr(srl_state, "current_step_count", 0)
                                    # Per-LAYER cache, as MLX keys its entry on layer_idx. A
                                    # single shared slot would hold whatever the LAST layer chose
                                    # and hand it to all the others.
                                    _rt_cache = getattr(srl_state, "_route_layer_cache", None)
                                    if _rt_cache is None:
                                        _rt_cache = srl_state._route_layer_cache = {}
                                    _rt_ent = _rt_cache.get(captured_layer_idx)
                                    _n_cand = int(block_indices.numel()) if block_indices is not None else 0
                                    # Same three conditions as MLX: never routed, interval
                                    # elapsed, or the candidate count changed (a block was
                                    # flushed into the pool since the last route, so its content
                                    # is currently attended nowhere).
                                    _should_route = (
                                        _rt_ent is None
                                        or (_step_ctr - _rt_ent[0]) >= srl_state._route_cadence
                                        or _rt_ent[1] != _n_cand
                                    )

                                    if _should_route:
                                        # Route at layer 0 — cache result for all 28 layers.
                                        # DKV_ROUTER selects the scorer:
                                        #   "residual" (default) — MLX-parity pure q·k relevance
                                        #     top-K over anchors + exact residual keys.  Zero host
                                        #     syncs; the router behind MLX's flat decode tps.
                                        #   "srl" — the legacy multi-channel router (lexical index
                                        #     + semantic ANN + chunk graph + anchor rerank).
                                        #     Measured net-negative on whole-document synthesis at
                                        #     13.4K (degraded outputs, lower tps); kept for
                                        #     multi-turn experiments.
                                        q_for_routing = query_states[b_idx, :, 0, :]  # [H, D] ROTATED query (matches MLX _block_relevance_residual)
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        _router_mode = os.environ.get("DKV_ROUTER", "residual").lower()
                                        if _router_mode == "srl":
                                            selected_slots = route_query_fixed_k(
                                                Q         = unrot_query_states[b_idx, :, 0, :],
                                                srl_state = srl_state,
                                                pool      = pool,
                                                scale     = _scale,
                                                layer_idx = captured_layer_idx,
                                            )
                                        else:
                                            from native_core.srl.query_router import route_blocks_relevance
                                            session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                                            cos_all = session_dict.get("rope_cos")
                                            sin_all = session_dict.get("rope_sin")
                                            _max_anc = int(anchor_indices.max().item()) + 1 if (anchor_indices is not None and anchor_indices.numel() > 0) else 1
                                            if pool is not None and getattr(pool, "residual_K_positions", None) is not None and pool.residual_K_positions.numel() > 0:
                                                _max_anc = max(_max_anc, int(pool.residual_K_positions.max().item()) + 1)
                                            if cos_all is None or sin_all is None or cos_all.shape[1] < _max_anc:
                                                hist_pos = torch.arange(_max_anc, device=query_states.device, dtype=torch.long).unsqueeze(0)
                                                _rot_emb = getattr(self, "rotary_emb", None) or getattr(getattr(model, "model", None), "rotary_emb", None)
                                                cos_all, sin_all = _rot_emb(value_states[b_idx:b_idx+1], hist_pos)
                                                session_dict["rope_cos"] = cos_all
                                                session_dict["rope_sin"] = sin_all
                                            selected_slots = route_blocks_relevance(
                                                Q              = q_for_routing,
                                                pool           = pool,
                                                block_indices  = block_indices,
                                                anchor_indices = anchor_indices,
                                                scale          = _scale,
                                                cos            = cos_all,
                                                sin            = sin_all,
                                                srl_state      = srl_state,
                                            )
                                        srl_state.current_step_slots = selected_slots

                                        # Map slot IDs to absolute sequence anchor indices.
                                        # argmax(dim=1) returns 0 for a row with NO match, so a
                                        # selected slot that is not among this layer's blocks would
                                        # silently map to block 0 — duplicating the sink and dropping
                                        # the intended block.  Keep only rows that actually matched.
                                        mask = (selected_slots.unsqueeze(1) == block_indices.unsqueeze(0))
                                        has_match = mask.any(dim=1)
                                        # Same two-syncs-on-one-predicate pattern as the
                                        # per-layer reroute below; resolve the shape once.
                                        _keep0 = torch.nonzero(has_match, as_tuple=True)[0]
                                        block_idx_in_full = mask.to(torch.uint8).argmax(dim=1)[_keep0]
                                        selected_anchors = anchor_indices[block_idx_in_full]
                                        srl_state.current_step_slots = selected_slots[_keep0]
                                        srl_state.current_step_anchors = selected_anchors
                                        # Publish to the PER-LAYER cache so the hold branch
                                        # below replays THIS layer's decision, not whichever
                                        # layer happened to route last.
                                        _rt_cache[captured_layer_idx] = (
                                            _step_ctr, _n_cand,
                                            selected_slots[_keep0], selected_anchors)

                                        # DKV_ROUTE_PROBE=2: report WHICH blocks
                                        # were kept, once per (layer, candidate
                                        # count). "Routing ran" does not answer
                                        # the question that matters for a deep
                                        # needle -- whether the block HOLDING it
                                        # survived top-K. At 32k the router keeps
                                        # 16 of ~122 blocks, and a depth-0.9
                                        # needle sits near anchor ~29300, so this
                                        # is a direct read of whether it was
                                        # dropped rather than an inference from
                                        # the model saying "None".
                                        if os.environ.get("DKV_ROUTE_PROBE", "1") == "2":
                                            _sp = getattr(kv_manager, "_route_sel_seen", None)
                                            if _sp is None:
                                                _sp = kv_manager._route_sel_seen = set()
                                            _sp_key = (sid, captured_layer_idx,
                                                       int(block_indices.numel()))
                                            if _sp_key not in _sp:
                                                _sp.add(_sp_key)
                                                _sa = selected_anchors.tolist()
                                                _all_a = anchor_indices.tolist()
                                                print(f"[DKV] ROUTE SELECTION layer={captured_layer_idx} "
                                                      f"kept {len(_sa)} of {len(_all_a)} blocks | "
                                                      f"anchor span kept "
                                                      f"[{min(_sa) if _sa else '-'}..{max(_sa) if _sa else '-'}] "
                                                      f"of [{min(_all_a) if _all_a else '-'}.."
                                                      f"{max(_all_a) if _all_a else '-'}] | "
                                                      f"kept={sorted(_sa)}", flush=True)
                                        selected_slots = srl_state.current_step_slots
                                    else:
                                        # HOLD this layer's own last decision, as MLX holds its
                                        # per-layer cache entry for the whole interval. Reading
                                        # the shared current_step_* here would replay whichever
                                        # layer routed most recently.
                                        selected_slots = _rt_ent[2]
                                        selected_anchors = _rt_ent[3]

                                else:
                                    # Other layers: reuse the layer-0 slot selection (default;
                                    # set DKV_ROUTE_PER_LAYER=1 above to route every layer
                                    # independently instead).
                                    selected_slots = getattr(srl_state, "current_step_slots", None)
                                    selected_anchors = getattr(srl_state, "current_step_anchors", None)

                                if (selected_slots is not None and selected_slots.numel() > 0
                                        and selected_anchors is not None and selected_anchors.numel() > 0):
                                    # 1. GPU mapping (for block_indices and anchor_indices used in kernels).
                                    # Same argmax(dim=1)==0 hazard as the slot→anchor map above: a
                                    # selected anchor absent from THIS layer (block-state can differ
                                    # per layer) would map to block 0.  Filter to matched anchors, and
                                    # only apply the reroute if at least one selected block survives —
                                    # otherwise fall through to full attention rather than collapsing
                                    # onto block 0.
                                    mask = (selected_anchors.unsqueeze(1) == anchor_indices.unsqueeze(0))
                                    has_match = mask.any(dim=1)
                                    # ONE sync, not three. bool(has_match.any()) and
                                    # each boolean-mask index below were three separate
                                    # device->host syncs on the SAME predicate -- the
                                    # sync recorder put them at 5,376 hits each, ~1 per
                                    # layer per decoded token, 26% of all syncs
                                    # combined. torch.nonzero resolves the shape once;
                                    # .numel() is then free and integer indexing is a
                                    # plain gather. x[bool_mask] and x[nonzero(mask)]
                                    # select the same rows in the same order.
                                    _keep = torch.nonzero(has_match, as_tuple=True)[0]
                                    if _keep.numel() > 0:
                                        block_idx_in_layer = mask.to(torch.uint8).argmax(dim=1)[_keep]
                                        block_indices = block_indices[block_idx_in_layer]
                                        anchor_indices = selected_anchors[_keep]

                                        # Cache is structured and self-evicting based on layer_idx and anchors_tuple comparison
                                        _srl_rerouted = True

                                    # Log routing decision if verbose or telemetry is enabled
                                    if captured_layer_idx == _first_dkv_layer and (os.environ.get("DKV_SRL_VERBOSE", "0") == "1" or os.environ.get("DKV_TELEMETRY", "0") == "1"):
                                        n_sel = selected_slots.numel()
                                        n_tot = srl_state.n_active_blocks()
                                        print(f"[SRL Decode Step] session={sid} step={srl_state.current_step_count} "
                                              f"selected={n_sel}/{n_tot} blocks (k_min={srl_state.k_min}, k_max={srl_state.k_max})")
                            except Exception as _srl_e:
                                # SRL failure is non-fatal — fall back to full attention
                                if os.environ.get("DKV_SRL_VERBOSE", "0") == "1":
                                    print(f"[SRL] route_query error: {_srl_e}")

                        # ── MLX-parity block pruning (DKV_DECODE_PRUNE_K) ──────
                        # Ports MLX's decode top-K: rank compressed blocks by exact
                        # q·k relevance (route_blocks_relevance == MLX
                        # _block_relevance_residual) and reconstruct only the top-K,
                        # instead of all ~49.  Routes once at layer 0 (caches the
                        # layer-invariant selected ANCHORS) and maps to each layer's
                        # slots.
                        #
                        # OFF BY DEFAULT (0) — CONFIRMED DEAD END on A100 (2026-07-18):
                        #   1. tps did NOT change at K=16 (49→16 blocks, decode 7.0
                        #      vs 7.0) — decode is bound by the eager nf4 model
                        #      forward, NOT block reconstruction, so pruning buys no
                        #      speed at this context (block count is irrelevant to
                        #      decode time here).
                        #   2. Output collapsed to garbage on every pruned preset —
                        #      the CUDA residual router drops answer-critical blocks
                        #      at K=16 where MLX's does not.
                        # Kept for the record; do not enable without a K that both
                        # preserves output AND a regime where reconstruction is the
                        # decode bottleneck (long context, not 13.4K).
                        _prune_k = int(os.environ.get("DKV_DECODE_PRUNE_K", "0"))
                        if (_prune_k > 0 and not _srl_rerouted and pool is not None
                                and block_indices is not None and anchor_indices is not None
                                and block_indices.numel() > _prune_k):
                            _ws = kv_manager.decode_workspace.setdefault(sid, {})
                            _pr = _ws.setdefault("_mlx_prune", {})
                            if captured_layer_idx == _first_dkv_layer:
                                try:
                                    from native_core.srl.query_router import route_blocks_relevance
                                    _q_prune = unrot_query_states[b_idx, :, 0, :]   # [H, D]
                                    _sc = 1.0 / math.sqrt(head_dim)
                                    _sel_slots = route_blocks_relevance(
                                        _q_prune, pool, block_indices, anchor_indices, _sc
                                    )
                                    _m = (block_indices.unsqueeze(1) == _sel_slots.unsqueeze(0)).any(dim=1)
                                    _pr["anchors"] = anchor_indices[_m]
                                except Exception:
                                    _pr["anchors"] = None
                            _sel_anchors = _pr.get("anchors")
                            if _sel_anchors is not None and _sel_anchors.numel() > 0:
                                _keep = (anchor_indices.unsqueeze(1) == _sel_anchors.unsqueeze(0)).any(dim=1)
                                if bool(_keep.any()):
                                    block_indices = block_indices[_keep]
                                    anchor_indices = anchor_indices[_keep]

                        # ── Increment routing version at layer 0 ──
                        if captured_layer_idx == _first_dkv_layer:
                            session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                            last_slots = session_dict.get("last_slots")
                            if block_indices is None:
                                changed = (last_slots is not None)
                            elif last_slots is None:
                                changed = True
                            else:
                                changed = not torch.equal(block_indices, last_slots)
                            if changed:
                                session_dict["last_slots"] = block_indices.clone() if block_indices is not None else None
                                current_version = session_dict.get("routing_version", 0) + 1
                                session_dict["routing_version"] = current_version
                                # Eagerly clear stale per-layer RoPE slice caches.
                                # When routing changes, previously cached decode_cos_sliced /
                                # decode_sin_sliced tensors for all 28 layers are invalid.
                                # Holding them wastes up to ~95MB VRAM until each layer
                                # lazily overwrites its own entry. One .clear() here frees
                                # the entire cache in O(1) before the next forward pass.
                                cos_c = session_dict.get("decode_cos_sliced")
                                sin_c = session_dict.get("decode_sin_sliced")
                                if cos_c is not None:
                                    cos_c.clear()
                                if sin_c is not None:
                                    sin_c.clear()

                        # ── Validation mode (DKV_VALIDATE_SRL=1) ───────────────
                        _validate_this_step = (
                            _SRL_VALIDATE
                            and _srl_rerouted
                            and captured_layer_idx == _first_dkv_layer
                            and srl_state is not None
                            and (srl_state.current_step_count % _SRL_VALIDATE_EVERY) == 0
                        )

                        # Assemble ONLY the dense window (non-compressed blocks) into a
                        # persistent fixed-size workspace tensor. Uses lightweight slice copies,
                        # zero GEMM. Compressed blocks handled by block_indices in sparse kernel.
                        # workspace shape is always [1, kv_heads, max_dense_len, head_dim];
                        # dense_len tells us how many positions are actually valid this step.
                        dense_k_assembled, dense_v_assembled, dense_len = None, None, 0
                        if dense_blocks:
                            dense_k_assembled, dense_v_assembled, dense_len, dense_blocks = kv_manager.assemble_dense_window_kv(
                                sid, captured_layer_idx, dense_blocks, query_states.dtype
                            )


                        total_seq_len = kv_manager.get_session_sequence_length(sid)
                        max_pos = total_seq_len
                        if max_anchor_idx is not None:
                            max_pos = max(max_pos, max_anchor_idx + session_mbs)
                        
                        session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                        cached_cos = session_dict.get("rope_cos")
                        cached_sin = session_dict.get("rope_sin")

                        if cached_cos is not None and cached_cos.shape[1] >= max_pos:
                            cos_all = cached_cos[:, :max_pos]
                            sin_all = cached_sin[:, :max_pos]
                        else:
                            hist_pos = torch.arange(max_pos, device=query_states.device, dtype=torch.long).unsqueeze(0)
                            _rot_emb_fn = _resolve_rotary_emb(model)
                            cos_all, sin_all = _rot_emb_fn(value_states[b_idx:b_idx+1], hist_pos)
                            session_dict["rope_cos"] = cos_all
                            session_dict["rope_sin"] = sin_all

                        cos_sliced_arg = None
                        sin_sliced_arg = None
                        if anchor_indices is not None and anchor_indices.numel() > 0 and pool is not None:
                            session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                            current_version = session_dict.get("routing_version", 0)
                            cos_sliced_cache = session_dict.setdefault("decode_cos_sliced", {})
                            sin_sliced_cache = session_dict.setdefault("decode_sin_sliced", {})
                            
                            cos_cached_val = cos_sliced_cache.get(captured_layer_idx)
                            sin_cached_val = sin_sliced_cache.get(captured_layer_idx)
                            
                            if cos_cached_val is not None and cos_cached_val[0] == current_version:
                                cos_sliced_cached = cos_cached_val[1]
                            else:
                                cos_sliced_cached = None
                            
                            if sin_cached_val is not None and sin_cached_val[0] == current_version:
                                sin_sliced_cached = sin_cached_val[1]
                            else:
                                sin_sliced_cached = None

                            if cos_sliced_cached is None or sin_sliced_cached is None:
                                N_blocks = anchor_indices.shape[0]
                                max_seq_len = pool.U.shape[1]
                                positions = anchor_indices.view(N_blocks, 1) + torch.arange(1 + max_seq_len, device=query_states.device).view(1, 1 + max_seq_len)
                                positions_flat = positions.view(-1)
                                
                                cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
                                sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all

                                seq_len_limit = cos_flat.shape[0]
                                positions_flat = positions_flat.clamp(min=0, max=seq_len_limit - 1).clone()
                                cos_gathered = cos_flat[positions_flat]
                                sin_gathered = sin_flat[positions_flat]
                                rotary_dim = cos_gathered.shape[-1]
                                if rotary_dim < head_dim:
                                    # Partial RoPE (Qwen3.5-style partial_rotary_factor<1.0): the
                                    # C++/ATen and Metal anchor-rotation kernels below always
                                    # rotate the full head_dim (no rotary_dim concept, fixed
                                    # head_dim//2 pairing) and read this buffer as [K, head_dim]
                                    # -- passing the true [K, rotary_dim] tensor would read past
                                    # its real width. Pad the tail with cos=1/sin=0 so those
                                    # dims come out unrotated (raw*1 + partner*0 == raw) instead
                                    # of reading out of bounds. The rotary sub-range itself still
                                    # uses the kernels' head_dim//2 pairing rather than the
                                    # mathematically-correct rotary_dim//2 one -- a known,
                                    # narrower approximation pending a native-kernel fix.
                                    _pad_shape = cos_gathered.shape[:-1] + (head_dim - rotary_dim,)
                                    cos_gathered = torch.cat(
                                        [cos_gathered, torch.ones(_pad_shape, device=cos_gathered.device, dtype=cos_gathered.dtype)], dim=-1)
                                    sin_gathered = torch.cat(
                                        [sin_gathered, torch.zeros(_pad_shape, device=sin_gathered.device, dtype=sin_gathered.dtype)], dim=-1)
                                cos_sliced_cached = cos_gathered.view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                sin_sliced_cached = sin_gathered.view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                
                                cos_sliced_cache[captured_layer_idx] = (current_version, cos_sliced_cached)
                                sin_sliced_cache[captured_layer_idx] = (current_version, sin_sliced_cached)

                            cos_sliced_arg = cos_sliced_cached
                            sin_sliced_arg = sin_sliced_cached

                        # Compute per-slot anchor cos/sin for C++/Metal on-the-fly RoPE rotation.
                        # cos_sliced_arg shape: [K, 1+max_seq_len, 1, D] — index 0 on dim1 is the anchor,
                        # already padded to full D width (see its construction above) so the
                        # fixed-buffer Metal kernel never reads out of bounds. We need [K, D]
                        # float32 for the C++/Metal kernel, PLUS the true (pre-padding) rotary
                        # width for the shader's rotary_dim param (it can't infer width from a
                        # buffer size the way a shaped tensor can).
                        _cos_anc_2d = None
                        _sin_anc_2d = None
                        _anchor_rotary_dim = cos_all.shape[-1]
                        if cos_sliced_arg is not None and cos_sliced_arg.numel() > 0:
                            _cos_anc_2d = cos_sliced_arg[:, 0, 0, :].to(torch.float32).contiguous()  # [K, D]
                            _sin_anc_2d = sin_sliced_arg[:, 0, 0, :].to(torch.float32).contiguous()  # [K, D]
                        # ATen kernels (decode_attention_aten*) derive rotary_dim from the cos/sin
                        # tensor's own last-dim size rather than a separate param, so they need
                        # the TRUE unpadded width, not the Metal-oriented padded buffer above --
                        # recover it by slicing the padding back off.
                        _cos_anc_2d_true = _cos_anc_2d[:, :_anchor_rotary_dim] if _cos_anc_2d is not None else None
                        _sin_anc_2d_true = _sin_anc_2d[:, :_anchor_rotary_dim] if _sin_anc_2d is not None else None

                        # ── Query Factual Store (Solution 4) ──
                        # Descriptors are built from layer-0 K vectors (factual_store.py:200,
                        # span_K[0]). Running the query at layers 1-27 compares against a
                        # different projection space and accumulates spurious cross-category
                        # matches into the logit-bias state. All state updates therefore run
                        # only at layer 0; layers 1-27 reuse the cached FactEntry list so
                        # the three-way attention combination still fires at every layer.
                        factual_store = getattr(kv_manager, "_factual_stores", {}).get(sid)
                        matching_entries = []
                        if factual_store is not None and pool is not None and pool.W_proj is not None:
                            try:
                                if captured_layer_idx == _first_dkv_layer:
                                    # ── Layer 0: run query and update all logit-bias state ──────
                                    active_slots = set(block_indices.tolist()) if block_indices is not None else None
                                    # Query Anchor Blending (layer-0 Q space only — blending a
                                    # layer-0 anchor with layer-N raw_q is semantically incorrect).
                                    raw_q = unrot_query_states[b_idx, :, 0, :]  # [H, D]
                                    if srl_state is not None:
                                        srl_state.current_step_factual_tokens = set()
                                        srl_state.current_step_factual_sequences = []
                                        srl_state.current_step_max_similarity = 0.0
                                        srl_state.current_step_sequence_entity_ids = []
                                        srl_state.current_step_sequence_is_prime = []
                                        srl_state.current_step_sequence_prefixes = []
                                        if srl_state.factual_anchor_q is None:
                                            srl_state.factual_anchor_q = raw_q.detach().clone()

                                            # ── Early Entity Binding (Component 4) ───────────────
                                            # Analyze the query tokens against prime entries at the
                                            # very start of generation.  This sets entity context
                                            # immediately instead of waiting 30+ generated tokens
                                            # for the Contrastive Category Anchor to activate.
                                            try:
                                                query_toks = set(getattr(srl_state, "current_query_tokens", []))
                                                inv_index = getattr(srl_state, "inverted_index", None)
                                                if query_toks and factual_store is not None and inv_index is not None:
                                                    important_query_toks = query_toks & inv_index.important_vocab
                                                    prime_matches = []
                                                    for fe in factual_store.entries:
                                                        if getattr(fe, "is_prime", False):
                                                            fe_important = set(fe.tokens) & inv_index.important_vocab
                                                            overlap = len(important_query_toks & fe_important)
                                                            if overlap >= 1:
                                                                prime_matches.append((fe.start_idx, overlap))
                                                    if len(prime_matches) == 1:
                                                        # Single-entity query: lock immediately
                                                        srl_state.current_entity_id = prime_matches[0][0]
                                                        srl_state.dual_entity_mode = False
                                                    elif len(prime_matches) >= 2:
                                                        # Comparison query: activate dual-entity mode
                                                        prime_matches.sort(key=lambda x: x[1], reverse=True)
                                                        srl_state.dual_entity_mode = True
                                                        srl_state.dual_entity_ids = [
                                                            prime_matches[0][0],
                                                            prime_matches[1][0],
                                                        ]
                                                        # RC5: sequence the comparison as per-entity
                                                        # blocks and lock to the first entity now,
                                                        # instead of leaving entity context open
                                                        # (which lets the two interleave).
                                                        srl_state.comparison_entities = list(srl_state.dual_entity_ids)
                                                        srl_state.comparison_active_idx = 0
                                                        srl_state.comparison_covered = set()
                                                        srl_state.current_entity_id = srl_state.comparison_entities[0]
                                            except Exception:
                                                pass  # Early binding is best-effort

                                        q_for_factual = 0.20 * raw_q + 0.80 * srl_state.factual_anchor_q.to(raw_q.device)
                                    else:
                                        q_for_factual = raw_q

                                    # RC3 — tell the store which entities the query
                                    # is about so it can rank their spans above
                                    # shared-vocabulary spans of other entities.
                                    _qbias = None
                                    if getattr(srl_state, "dual_entity_mode", False) and getattr(srl_state, "dual_entity_ids", None):
                                        _qbias = set(srl_state.dual_entity_ids)
                                    elif getattr(srl_state, "current_entity_id", -1) != -1:
                                        _qbias = {srl_state.current_entity_id}

                                    matching_entries = factual_store.query(
                                        Q=q_for_factual,
                                        W_proj=pool.W_proj,
                                        threshold=0.50,
                                        active_slots=active_slots,
                                        query_entity_bias=_qbias,
                                    )
                                    if srl_state is not None and matching_entries:
                                        for entry in matching_entries:
                                            srl_state.current_step_factual_tokens.update(entry.tokens)
                                            if entry.tokens not in srl_state.current_step_factual_sequences:
                                                srl_state.current_step_factual_sequences.append(entry.tokens)
                                            # RC1 — inject triple sequences from prime entries.
                                            # Triple sequences = (bridge + value) pairs grounded in
                                            # the source text.  By adding them here, the VSL can lock
                                            # onto the complete (relation → value) ordering, preventing
                                            # the model from freely substituting its own connectives.
                                            if getattr(entry, "is_prime", False):
                                                for triple_seq in getattr(entry, "triple_sequences", []):
                                                    if triple_seq and triple_seq not in srl_state.current_step_factual_sequences:
                                                        srl_state.current_step_factual_sequences.append(triple_seq)
                                                        srl_state.current_step_factual_tokens.update(triple_seq)
                                            # ── 1-hop neighbor injection ────────────────────────────
                                            for nb_idx, nb_weight in zip(entry.neighbors, entry.weights):
                                                if nb_weight >= 0.45 and nb_idx < len(factual_store.entries):
                                                    nb_e = factual_store.entries[nb_idx]
                                                    if nb_e.tokens and nb_e.tokens not in srl_state.current_step_factual_sequences:
                                                        srl_state.current_step_factual_tokens.update(nb_e.tokens)
                                                        srl_state.current_step_factual_sequences.append(nb_e.tokens)
                                                    if getattr(nb_e, "is_prime", False):
                                                        for triple_seq in getattr(nb_e, "triple_sequences", []):
                                                            if triple_seq and triple_seq not in srl_state.current_step_factual_sequences:
                                                                srl_state.current_step_factual_sequences.append(triple_seq)
                                                                srl_state.current_step_factual_tokens.update(triple_seq)
                                                        # ── 2-hop neighbor injection ────────────────
                                                    for nb2_idx, nb2_weight in zip(nb_e.neighbors, nb_e.weights):
                                                            if nb2_weight >= 0.65 and nb2_idx < len(factual_store.entries):
                                                                nb2_e = factual_store.entries[nb2_idx]
                                                                if nb2_e.tokens and nb2_e.tokens not in srl_state.current_step_factual_sequences:
                                                                    srl_state.current_step_factual_tokens.update(nb2_e.tokens)
                                                                    srl_state.current_step_factual_sequences.append(nb2_e.tokens)
                                        sims = [getattr(e, "current_sim", 0.0) for e in matching_entries]
                                        if sims:
                                            srl_state.current_step_max_similarity = max(srl_state.current_step_max_similarity, max(sims))

                                    # ── Lexical Tripwire ────────────────────────────────────────────
                                    if srl_state is not None and factual_store is not None:
                                        recent_toks = getattr(srl_state, "recent_generated_tokens", [])
                                        inv_index = getattr(srl_state, "inverted_index", None)
                                        if recent_toks and inv_index is not None:
                                            last_tok = recent_toks[-1]
                                            tok_idf = inv_index.idf.get(last_tok, 0.0)
                                            if tok_idf >= 2.5 and last_tok in inv_index.occurrences:
                                                occ_slots = set(occ[0] for occ in inv_index.occurrences[last_tok])
                                                for fe in factual_store.entries:
                                                    if last_tok in fe.tokens and any(s in occ_slots for s in fe.slot_ids):
                                                        if fe.tokens not in srl_state.current_step_factual_sequences:
                                                            srl_state.current_step_factual_tokens.update(fe.tokens)
                                                            srl_state.current_step_factual_sequences.append(fe.tokens)
                                                        for nb_idx, nb_weight in zip(fe.neighbors, fe.weights):
                                                            if nb_weight >= 0.45 and nb_idx < len(factual_store.entries):
                                                                nb_e = factual_store.entries[nb_idx]
                                                                if nb_e.tokens and nb_e.tokens not in srl_state.current_step_factual_sequences:
                                                                    srl_state.current_step_factual_tokens.update(nb_e.tokens)
                                                                    srl_state.current_step_factual_sequences.append(nb_e.tokens)
                                                        break

                                    # ── Contrastive Category Anchor ──────────────────────────────────
                                    # For single-category questions: fires when exactly ONE prime
                                    # matches recent tokens — filters sequences to ±512 positions.
                                    # For comparison questions (two primes active): fires when one
                                    # prime has ≥2× the recent-token overlap of the other — the model
                                    # is actively generating about one category, so lock to it.
                                    # Without this extension, comparison questions NEVER trigger the
                                    # anchor and both categories' tokens stay merged in the VSL set.
                                    if srl_state is not None and factual_store is not None:
                                        inv_index = getattr(srl_state, "inverted_index", None)
                                        recent_set = set(getattr(srl_state, "recent_generated_tokens", [])[-30:])
                                        best_prime_pos = None
                                        best_overlap = 0
                                        second_best_overlap = 0
                                        active_prime_count = 0

                                        for fe in factual_store.entries:
                                            if getattr(fe, "is_prime", False):
                                                if inv_index is not None:
                                                    important_recent = recent_set & inv_index.important_vocab
                                                    fe_important = set(fe.tokens) & inv_index.important_vocab
                                                    overlap = len(important_recent & fe_important)
                                                else:
                                                    overlap = sum(1 for t in fe.tokens if t in recent_set)
                                                if overlap >= 1:
                                                    active_prime_count += 1
                                                    if overlap > best_overlap:
                                                        second_best_overlap = best_overlap
                                                        best_overlap = overlap
                                                        best_prime_pos = fe.start_idx
                                                    elif overlap > second_best_overlap:
                                                        second_best_overlap = overlap

                                        # Determine effective prime: single active OR dominant in
                                        # two-prime comparison (one has ≥2× the other's overlap).
                                        effective_prime_pos = None
                                        if active_prime_count == 1 and best_prime_pos is not None:
                                            effective_prime_pos = best_prime_pos
                                        elif (active_prime_count == 2 and best_prime_pos is not None
                                              and best_overlap >= 2
                                              and best_overlap >= 2 * (second_best_overlap + 1)):
                                            effective_prime_pos = best_prime_pos

                                        if effective_prime_pos is not None:
                                            # Update entity context: the CA has identified the dominant entity.
                                            # This kicks in ASAP (doesn't wait for 30+ generated tokens) so
                                            # early-generation entity binding is also enforced.
                                            srl_state.current_entity_id = effective_prime_pos
                                            if not getattr(srl_state, "dual_entity_mode", False):
                                                pos_map = {tuple(fe.tokens): fe.start_idx for fe in factual_store.entries}
                                                filtered_seqs = []
                                                for s in srl_state.current_step_factual_sequences:
                                                    fe_pos = pos_map.get(tuple(s))
                                                    if fe_pos is None or abs(fe_pos - effective_prime_pos) < 512:
                                                        filtered_seqs.append(s)
                                                if filtered_seqs:
                                                    srl_state.current_step_factual_sequences = filtered_seqs
                                                    srl_state.current_step_factual_tokens = set()
                                                    for s in filtered_seqs:
                                                        srl_state.current_step_factual_tokens.update(s)

                                        # ── RC5 Comparison Sequencing ───────────────────────────
                                        # In comparison mode the anchor's job is segmentation, not
                                        # winner-selection: lock to one entity's block, advance only
                                        # once it is substantively covered.  This overrides whatever
                                        # the dominance heuristic above guessed.
                                        if (getattr(srl_state, "dual_entity_mode", False)
                                                and getattr(srl_state, "comparison_entities", None)):
                                            from native_core.srl.factual_alignment import advance_comparison_entity
                                            prime_tok_by_ent: Dict = {}
                                            prop_tok_by_ent: Dict = {}
                                            for fe in factual_store.entries:
                                                eid = getattr(fe, "entity_id", -1)
                                                if eid == -1:
                                                    continue
                                                if getattr(fe, "is_prime", False):
                                                    prime_tok_by_ent.setdefault(eid, set()).update(fe.tokens)
                                                else:
                                                    prop_tok_by_ent.setdefault(eid, set()).update(fe.tokens)
                                            new_idx, new_cov = advance_comparison_entity(
                                                srl_state.comparison_entities,
                                                getattr(srl_state, "comparison_active_idx", 0),
                                                getattr(srl_state, "comparison_covered", set()),
                                                getattr(srl_state, "recent_generated_tokens", [])[-30:],
                                                prime_tok_by_ent,
                                                prop_tok_by_ent,
                                            )
                                            srl_state.comparison_active_idx = new_idx
                                            srl_state.comparison_covered = new_cov
                                            srl_state.current_entity_id = srl_state.comparison_entities[new_idx]
                                        # ── Coherence Cap (RC6 entity-proportional) ─────────────
                                    if srl_state is not None and factual_store is not None:
                                        n_active_primes = sum(
                                            1 for fe in factual_store.entries
                                            if getattr(fe, "is_prime", False) and getattr(fe, "current_sim", 0.0) > 0
                                        )
                                        # 4 sequences per active entity + 4 overhead; minimum 8.
                                        # This prevents one entity from monopolising the budget
                                        # in comparison questions.
                                        coherence_cap = max(8, n_active_primes * 4 + 4)
                                        if len(srl_state.current_step_factual_sequences) > coherence_cap:
                                            seq_id_to_score = {}
                                            for fe_e in factual_store.entries:
                                                fe_sim = getattr(fe_e, "current_sim", 0.0)
                                                if fe_sim > 0:
                                                    seq_id_to_score[tuple(fe_e.tokens)] = max(
                                                        seq_id_to_score.get(tuple(fe_e.tokens), 0.0), fe_sim
                                                    )
                                            # DX1: triple sequences inherit their prime's score so
                                            # they sort above tangential context and aren't evicted.
                                            for fe_e in factual_store.entries:
                                                if getattr(fe_e, "is_prime", False):
                                                    prime_sim = getattr(fe_e, "current_sim", 0.0)
                                                    if prime_sim > 0:
                                                        for ts in getattr(fe_e, "triple_sequences", []):
                                                            tup = tuple(ts)
                                                            if tup not in seq_id_to_score:
                                                                seq_id_to_score[tup] = prime_sim
                                            srl_state.current_step_factual_sequences.sort(
                                                key=lambda s: seq_id_to_score.get(tuple(s), 0.0), reverse=True
                                            )
                                            srl_state.current_step_factual_sequences = srl_state.current_step_factual_sequences[:coherence_cap]
                                            srl_state.current_step_factual_tokens = set()
                                            for s in srl_state.current_step_factual_sequences:
                                                srl_state.current_step_factual_tokens.update(s)

                                    # ── Entity-Subgraph Tagging ───────────────────────────────────
                                    # Use the entity_id from factual_store.build() (token-overlap
                                    # matching — not positional proximity, which is wrong for
                                    # interleaved comparison text).  Triple sequences (bridge +
                                    # value) inherit the prime's entity_id since they were extracted
                                    # from that prime's context.
                                    if srl_state is not None and factual_store is not None:
                                        # Build lookup: (entry.tokens as tuple) → (entity_id, is_prime, prefix_tokens)
                                        entry_meta: Dict = {}
                                        for fe in factual_store.entries:
                                            tup = tuple(fe.tokens)
                                            entry_meta[tup] = (
                                                getattr(fe, "entity_id", -1),
                                                getattr(fe, "is_prime", False),
                                                getattr(fe, "prefix_tokens", []),
                                            )
                                        # Also build lookup for triple sequences owned by each prime
                                        triple_to_entity: Dict = {}
                                        for fe in factual_store.entries:
                                            if getattr(fe, "is_prime", False):
                                                p_entity = getattr(fe, "entity_id", -1)
                                                for ts in getattr(fe, "triple_sequences", []):
                                                    triple_to_entity[tuple(ts)] = p_entity

                                        entity_ids = []
                                        is_prime_flags = []
                                        seq_prefixes = []
                                        for seq in srl_state.current_step_factual_sequences:
                                            tup = tuple(seq)
                                            if tup in entry_meta:
                                                eid, isp, pref = entry_meta[tup]
                                            elif tup in triple_to_entity:
                                                # Triple sequence — inherits prime's entity.  Its
                                                # bridge connective is already in the sequence, so
                                                # it needs no external prefix grounding (RC2).
                                                eid, isp, pref = triple_to_entity[tup], False, []
                                            else:
                                                eid, isp, pref = -1, False, []
                                            entity_ids.append(eid)
                                            is_prime_flags.append(isp)
                                            seq_prefixes.append(list(pref))
                                        srl_state.current_step_sequence_entity_ids = entity_ids
                                        srl_state.current_step_sequence_is_prime = is_prime_flags
                                        srl_state.current_step_sequence_prefixes = seq_prefixes

                                    # Cache entries so layers 1-27 can run K/V attention without
                                    # re-querying or touching logit-bias state.
                                    if srl_state is not None:
                                        srl_state._cached_factual_entries = matching_entries

                                else:
                                    # Layers 1-27: reuse the layer-0 FactEntry list for the
                                    # three-way K/V attention combination only. entry.K[layer_idx]
                                    # gives the correct per-layer keys; no state is mutated here.
                                    if srl_state is not None:
                                        matching_entries = getattr(srl_state, "_cached_factual_entries", [])

                            except Exception as fe:
                                print(f"[SRL] WARNING: Factual store query failed: {fe}")

                        if matching_entries:
                            # 1. Dense recency attention output & LSE
                            has_dense = (dense_k_assembled is not None and dense_len > 0)
                            H_q = query_states.shape[1]
                            D = query_states.shape[3]
                            if has_dense:
                                dense_positions_list = []
                                for blk in dense_blocks:
                                    dense_positions_list.extend(blk.token_indices)
                                dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                cos_dense = cos_all[0, dense_positions.clamp(min=0, max=cos_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)
                                sin_dense = sin_all[0, dense_positions.clamp(min=0, max=sin_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)
                                # Slice to dense_len for correct shapes (cos_dense has shape [dense_len, D])
                                dense_k_valid = dense_k_assembled[:, :, :dense_len]
                                dense_v_valid = dense_v_assembled[:, :, :dense_len]
                                # Partial RoPE: cos_dense/sin_dense are rotary_dim wide,
                                # which is < head_dim on Qwen3.5 (64 vs 256). Pair within
                                # [0, rotary_dim) and pass the tail through unrotated;
                                # head_dim//2 pairing here both picked the wrong partner
                                # and raised on the cos multiply.
                                rot_dim = cos_dense.shape[-1]
                                half_d = rot_dim // 2
                                dense_k_half = torch.zeros_like(dense_k_valid[..., :rot_dim])
                                dense_k_half[..., :half_d] = -dense_k_valid[..., half_d:rot_dim]
                                dense_k_half[..., half_d:] = dense_k_valid[..., :half_d]

                                dense_k_rot = dense_k_valid.clone()
                                dense_k_rot[..., :rot_dim] = (
                                    dense_k_valid[..., :rot_dim] * cos_dense
                                    + dense_k_half * sin_dense
                                )
                                
                                k_rep = repeat_kv(dense_k_rot, num_key_value_groups)
                                v_rep = repeat_kv(dense_v_valid, num_key_value_groups)
                                out_dense = F.scaled_dot_product_attention(
                                    query_states[b_idx:b_idx+1], k_rep, v_rep,
                                    is_causal=False,
                                )  # [1, H_q, 1, D]
                                out_dense_hd = out_dense[0, :, 0, :].float()
                                
                                _q = query_states[b_idx, :, 0, :]
                                _kd = k_rep[0]
                                _scale = (D ** -0.5)
                                scores_dense = torch.matmul(_kd, _q.unsqueeze(-1)).squeeze(-1) * _scale  # [H_q, T]
                                lse_dense = torch.logsumexp(scores_dense.float(), dim=-1)  # [H_q]
                            else:
                                out_dense_hd = torch.zeros((H_q, D), dtype=torch.float32, device=query_states.device)
                                lse_dense = torch.full((H_q,), float('-inf'), dtype=torch.float32, device=query_states.device)

                            # 2. Sparse semantic attention output & LSE
                            has_comp = (block_indices is not None and block_indices.numel() > 0)
                            if has_comp:
                                _Q_sq = query_states[b_idx, :, 0, :]
                                _bsizes = pool.seq_lens[block_indices]
                                out_sparse, lse_sparse = fused_decode_mps(
                                    Q                    = _Q_sq,
                                    pool                 = pool,
                                    block_indices        = block_indices,
                                    blk_sizes            = _bsizes,
                                    num_key_value_groups = num_key_value_groups,
                                    anchor_indices       = anchor_indices,
                                    cos                  = cos_all,
                                    sin                  = sin_all,
                                )
                                out_sparse_fp32 = out_sparse.float()
                                lse_sparse = lse_sparse.to(torch.float32)
                            else:
                                out_sparse_fp32 = torch.zeros((H_q, D), dtype=torch.float32, device=query_states.device)
                                lse_sparse = torch.full((H_q,), float('-inf'), dtype=torch.float32, device=query_states.device)

                            # 3. Factual store matches attention output & LSE
                            fact_k_list = []
                            fact_v_list = []
                            fact_positions = []
                            for entry in matching_entries:
                                k_layer = entry.K[captured_layer_idx].to(device=query_states.device, dtype=query_states.dtype)
                                v_layer = entry.V[captured_layer_idx].to(device=query_states.device, dtype=query_states.dtype)
                                fact_k_list.append(k_layer)
                                fact_v_list.append(v_layer)
                                fact_positions.extend(range(entry.start_idx, entry.end_idx))
                                
                            fact_k = torch.cat(fact_k_list, dim=1).unsqueeze(0) # [1, kv_heads, total_fact_len, D]
                            fact_v = torch.cat(fact_v_list, dim=1).unsqueeze(0) # [1, kv_heads, total_fact_len, D]
                            
                            pos_tensor = torch.tensor(fact_positions, dtype=torch.long, device=query_states.device)
                            cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
                            sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
                            if cos_flat.dim() == 3:
                                cos_flat = cos_flat.squeeze(0)
                            if sin_flat.dim() == 3:
                                sin_flat = sin_flat.squeeze(0)
                                
                            pos_clamped = pos_tensor.clamp(min=0, max=cos_flat.shape[0] - 1)
                            cos_fact = cos_flat[pos_clamped].unsqueeze(0)
                            sin_fact = sin_flat[pos_clamped].unsqueeze(0)
                            
                            _, fact_k_rot = apply_rotary_pos_emb(
                                q=query_states[b_idx:b_idx+1],
                                k=fact_k,
                                cos=cos_fact,
                                sin=sin_fact
                            )
                            
                            k_rep_fact = repeat_kv(fact_k_rot, num_key_value_groups)
                            v_rep_fact = repeat_kv(fact_v, num_key_value_groups)
                            
                            _scale = D ** -0.5
                            out_facts = F.scaled_dot_product_attention(
                                query_states[b_idx:b_idx+1], k_rep_fact, v_rep_fact,
                                is_causal=False,
                                scale=_scale
                            )
                            out_facts_hd = out_facts[0, :, 0, :].float()
                            
                            _q = query_states[b_idx, :, 0, :]
                            _kf = k_rep_fact[0]
                            scores_fact = torch.matmul(_kf, _q.unsqueeze(-1)).squeeze(-1) * _scale
                            lse_facts = torch.logsumexp(scores_fact.float(), dim=-1)
                            lse_facts = lse_facts.to(torch.float32)

                            # Apply Factual LSE Attention Boosting
                            max_sim = max([getattr(entry, "current_sim", 0.0) for entry in matching_entries]) if matching_entries else 0.0
                            if max_sim >= 0.4:
                                boost = 8.0 * (max_sim - 0.4) / 0.6
                                lse_facts = lse_facts + boost

                            # 4. Three-way LSE combination (unclamped to preserve true log-sum-exp combination math)
                            # NaN guards (not just Inf, MLX parity): torch.maximum propagates NaN from
                            # either operand, so a single NaN in any of the three lse_* tensors would
                            # otherwise poison lse_max_masked -> every w_* -> the whole merge for that
                            # row. Treat NaN the same as Inf (contribute zero weight) at each guard.
                            lse_sparse = _apply_sparse_bias(lse_sparse, lse_dense)
                            lse_max = torch.maximum(torch.maximum(lse_dense, lse_sparse), lse_facts)
                            lse_max_masked = lse_max.clone()
                            lse_max_masked[~torch.isfinite(lse_max)] = 0.0

                            w_dense = torch.exp(lse_dense - lse_max_masked)
                            w_sparse = torch.exp(lse_sparse - lse_max_masked)
                            w_facts = torch.exp(lse_facts - lse_max_masked)

                            w_dense[~torch.isfinite(lse_dense)] = 0.0
                            w_sparse[~torch.isfinite(lse_sparse)] = 0.0
                            w_facts[~torch.isfinite(lse_facts)] = 0.0

                            denom = w_dense + w_sparse + w_facts
                            denom_safe = torch.clamp(denom, min=1e-9)

                            out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                                         out_sparse_fp32 * w_sparse.unsqueeze(-1) +
                                         out_facts_hd * w_facts.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
                            # Final safety net (MLX parity): zero any NaN that still made it through
                            # (e.g. from a NaN in an attention output itself, not just the lse weights).
                            out_final = torch.nan_to_num(out_final, nan=0.0)
                            attn_out_b = out_final.to(query_states.dtype).unsqueeze(0).unsqueeze(2)
                            attn_outputs.append(attn_out_b)
                            continue

                        # ── MPS Fast Path (Phase 34): fused_decode_mps ─────────────────────────
                        # On MPS (Apple Silicon), avoid _pytorch_vectorized_sparse_attn_decode
                        # which reconstructs all compressed K tokens from SVD + applies RoPE,
                        # causing ~180ms/tok overhead on long contexts.
                        #
                        # fused_decode_mps uses Project-Then-Attend (no per-token RoPE on
                        # compressed blocks) — valid approximation for far-away history tokens
                        # where content similarity dominates position alignment.
                        # Dense window tokens still receive exact pre-rotated attention.
                        _is_mps_decode = (query_states.device.type == "mps" and pool is not None and os.environ.get("DKV_MPS_APPROXIMATE_ATTN", "0") == "1")
                        if _is_mps_decode:
                            if _DKV_CORE_AVAILABLE and hasattr(_dkv_core, "fused_decode_attention_combined"):
                                _scale = 1.0 / math.sqrt(head_dim)
                                _q_val = query_states[b_idx, :, 0, :]
                                _dk = dense_k_assembled if dense_k_assembled is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                _dv = dense_v_assembled if dense_v_assembled is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                
                                if dense_k_assembled is not None:
                                    # OPT (P1-7): reuse cached position tensor (shared with CUDA combined path)
                                    _cache_key = (session_dict.get("routing_version", 0), dense_len)
                                    _dp_cache  = session_dict.get("dense_pos_tensor_cache")
                                    if _dp_cache is not None and _dp_cache[0] == _cache_key:
                                        dense_positions = _dp_cache[1].to(query_states.device)
                                    else:
                                        dense_positions_list = []
                                        for blk in dense_blocks:
                                            dense_positions_list.extend(blk.token_indices)
                                        dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                    # .to(torch.float32) is REQUIRED, not a nicety.
                                    #
                                    # dkv_decode.metal declares cos_dense/sin_dense as
                                    # `device const float*`. cos_all/sin_all carry the model's
                                    # dtype (fp16 here), so passing the slice through unconverted
                                    # made the shader reinterpret each PAIR of halves as one
                                    # float32 -- and read twice the buffer's byte length, running
                                    # off the end into whatever the allocator had there. Result:
                                    # garbage rotation angles, a handful of astronomically large
                                    # dense scores, and a softmax whose max was junk (global_d
                                    # collapsed to exactly 1 -- one outlier swamping ~2700 real
                                    # tokens). Because the overrun landed in recycled memory it
                                    # differed run to run, which is what made decode output
                                    # nondeterministic at temperature 0 on byte-identical inputs.
                                    # The anchor tables a few hundred lines up always did this
                                    # conversion (_cos_anc_2d), which is why only the dense half
                                    # of PASS 1 was corrupted.
                                    _cos = cos_all[0, dense_positions.clamp(min=0, max=cos_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1).to(torch.float32)
                                    _sin = sin_all[0, dense_positions.clamp(min=0, max=sin_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1).to(torch.float32)
                                    # Same Metal-buffer-width rationale as the anchor cos/sin
                                    # padding above: the shader indexes this as [.., head_dim],
                                    # not [.., rotary_dim].
                                    if _cos.shape[-1] < head_dim:
                                        _dense_pad_shape = _cos.shape[:-1] + (head_dim - _cos.shape[-1],)
                                        _cos = torch.cat([_cos, torch.ones(_dense_pad_shape, device=_cos.device, dtype=_cos.dtype)], dim=-1)
                                        _sin = torch.cat([_sin, torch.zeros(_dense_pad_shape, device=_sin.device, dtype=_sin.dtype)], dim=-1)
                                else:
                                    _cos = torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    _sin = torch.empty(0, device=query_states.device, dtype=torch.float32)

                                _slots = block_indices if block_indices is not None else torch.empty(0, device=query_states.device, dtype=torch.int32)
                                _ca = _cos_anc_2d if _cos_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                _sa = _sin_anc_2d if _sin_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)

                                _res_pos_K = pool.residual_K_positions if pool.residual_K_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                _res_val_K = pool.residual_K_values if pool.residual_K_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                _res_pos_V = pool.residual_V_positions if pool.residual_V_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                _res_val_V = pool.residual_V_values if pool.residual_V_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                _fact_pos = pool.fact_anchor_positions if pool.fact_anchor_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                _fact_val_K = pool.fact_anchors_K if pool.fact_anchors_K is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                _fact_val_V = pool.fact_anchors_V if pool.fact_anchors_V is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)

                                # DKV_LAYER_ADAPTIVE_RANK (default on) compresses this
                                # layer's blocks at get_layer_rank(...), not the flat
                                # kv_manager.rank -- passing the flat value here caps the
                                # kernel's delta reconstruction to fewer components than
                                # a boosted-rank layer (e.g. 32 of 48) actually stored,
                                # even after the pool_rank stride fix makes it read the
                                # RIGHT slot's data. Match what compression really used.
                                from native_core.kv_runtime_manager import get_layer_rank as _get_layer_rank
                                _cfg = getattr(kv_manager, "config", None)
                                _layer_active_rank = _get_layer_rank(
                                    captured_layer_idx, kv_manager.num_layers, kv_manager.rank,
                                    early_boost=getattr(_cfg, "early_layer_rank_boost", False),
                                    max_rank_early=getattr(_cfg, "max_rank_early", 0),
                                )

                                # Exact-position RoPE for residual/fact overrides
                                # (DKV_RESIDUAL_EXACT_ROPE, default on — matches the
                                # CUDA/Triton path and MLX). residual_K/fact_K hold the
                                # EXACT key of one specific token at absolute position
                                # anchor+offset; rotating them at the block anchor's angle
                                # instead scrambles precisely the position-sensitive
                                # digit/code tokens these overrides exist to preserve.
                                # Pass the model's raw full-sequence tables (row stride =
                                # rotary_dim, NOT padded to head_dim — the kernel is told
                                # the stride explicitly) plus each routed slot's absolute
                                # anchor position. cos_all/sin_all are already cached in
                                # the session workspace, so this is a pointer pass, not a
                                # gather. Empty tensors => kernel falls back to the old
                                # anchor-position approximation.
                                _exact_res_rope = os.environ.get("DKV_RESIDUAL_EXACT_ROPE", "1") == "1"
                                _empty_f32 = torch.empty(0, device=query_states.device, dtype=torch.float32)
                                if _exact_res_rope and anchor_indices is not None and anchor_indices.numel() > 0:
                                    _cf = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
                                    _sf = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
                                    _cos_full = _cf.to(torch.float32).contiguous()
                                    _sin_full = _sf.to(torch.float32).contiguous()
                                    _anchor_pos_i32 = anchor_indices.to(torch.int32).contiguous()
                                else:
                                    _cos_full = _empty_f32
                                    _sin_full = _empty_f32
                                    _anchor_pos_i32 = torch.empty(0, device=query_states.device, dtype=torch.int32)

                                _time_attn = os.environ.get("DKV_TIME_ATTN") == "1"
                                if _time_attn:
                                    import time as _t_mod
                                    if query_states.device.type == "mps":
                                        torch.mps.synchronize()
                                    _t_kernel_start = _t_mod.perf_counter()
                                out_val = _dkv_core.fused_decode_attention_combined(
                                    _q_val,
                                    _dk,
                                    _dv,
                                    _cos.contiguous(),
                                    _sin.contiguous(),
                                    pool.U,
                                    pool.U_scale,
                                    pool.V_K,
                                    pool.V_V,
                                    pool.anchors_K,
                                    pool.anchors_V,
                                    pool.seq_lens,
                                    pool.scales.contiguous(),
                                    _ca,
                                    _sa,
                                    _slots,
                                    _scale,
                                    num_heads,
                                    num_key_value_heads,
                                    _layer_active_rank,
                                    _res_pos_K.contiguous(),
                                    _res_val_K.contiguous(),
                                    _res_pos_V.contiguous(),
                                    _res_val_V.contiguous(),
                                    _fact_pos.contiguous(),
                                    _fact_val_K.contiguous(),
                                    _fact_val_V.contiguous(),
                                    _anchor_rotary_dim,
                                    _cos_full,
                                    _sin_full,
                                    _anchor_pos_i32,
                                )
                                if _time_attn:
                                    if query_states.device.type == "mps":
                                        torch.mps.synchronize()
                                    _t_kernel_ms = (_t_mod.perf_counter() - _t_kernel_start) * 1000
                                    print(f"[DKV_TIME_ATTN] fused_kernel={_t_kernel_ms:.2f}ms", flush=True)

                                attn_out_b = out_val.unsqueeze(0).unsqueeze(2)
                            else:
                                # ── Separate Dense SDPA and Compressed fused_decode_mps combined via LSE ──
                                has_dense = (dense_k_assembled is not None and dense_len > 0)
                                has_comp  = (block_indices is not None and block_indices.numel() > 0)
                                H_q       = query_states.shape[1]
                                D         = query_states.shape[3]

                                if has_dense:
                                    # 1. Optimize RoPE Slicing: gather non-contiguous positions directly
                                    # OPT (P1-7): cache dense position tensor across steps (same routing_version)
                                    _cache_key = (session_dict.get("routing_version", 0), dense_len)
                                    _dp_cache  = session_dict.get("dense_pos_tensor_cache")
                                    if _dp_cache is not None and _dp_cache[0] == _cache_key:
                                        dense_positions = _dp_cache[1].to(query_states.device)
                                    else:
                                        dense_positions_list = []
                                        for blk in dense_blocks:
                                            dense_positions_list.extend(blk.token_indices)
                                        dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                    L_dense = dense_positions.shape[0]
                                    cos_dense = cos_all[0, dense_positions.clamp(min=0, max=cos_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)
                                    sin_dense = sin_all[0, dense_positions.clamp(min=0, max=sin_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)

                                    # 2. Pre-allocate static workspace for dense_k_rot to eliminate dynamic shape allocations
                                    session_dict = kv_manager.decode_workspace.setdefault(sid, {})
                                    dense_k_rot_cache = session_dict.setdefault("dense_workspace_k_rot", {})
                                    workspace_k_rot = dense_k_rot_cache.get(captured_layer_idx)
                                    if (workspace_k_rot is None 
                                        or workspace_k_rot.shape[1] != num_key_value_heads 
                                        or workspace_k_rot.dtype != query_states.dtype 
                                        or workspace_k_rot.shape[2] < L_dense):
                                        alloc_len = ((L_dense + 511) // 512) * 512
                                        workspace_k_rot = torch.zeros((1, num_key_value_heads, alloc_len, head_dim), 
                                                                      device=query_states.device, dtype=query_states.dtype)
                                        dense_k_rot_cache[captured_layer_idx] = workspace_k_rot

                                    # 2b. Pre-allocate static workspace for dense_k_half to avoid rotate_half allocations
                                    dense_k_half_cache = session_dict.setdefault("dense_workspace_k_half", {})
                                    workspace_k_half = dense_k_half_cache.get(captured_layer_idx)
                                    if (workspace_k_half is None 
                                        or workspace_k_half.shape[1] != num_key_value_heads 
                                        or workspace_k_half.dtype != query_states.dtype 
                                        or workspace_k_half.shape[2] < L_dense):
                                        alloc_len = ((L_dense + 511) // 512) * 512
                                        workspace_k_half = torch.zeros((1, num_key_value_heads, alloc_len, head_dim), 
                                                                       device=query_states.device, dtype=query_states.dtype)
                                        dense_k_half_cache[captured_layer_idx] = workspace_k_half
                                    
                                    # Partial RoPE (Qwen3.5-style partial_rotary_factor<1.0):
                                    # cos_dense/sin_dense are only rotary_dim wide, not the
                                    # full head_dim -- rotate just that sub-range in the
                                    # pre-allocated workspace and copy the remainder through
                                    # unrotated. Reduces to the original full-width behavior
                                    # exactly when rotary_dim == head_dim.
                                    rotary_dim = cos_dense.shape[-1]
                                    half_d = rotary_dim // 2
                                    dense_k_half = workspace_k_half[:, :, :L_dense, :rotary_dim]
                                    dense_k_half[..., :half_d] = -dense_k_assembled[..., half_d:rotary_dim]
                                    dense_k_half[..., half_d:] = dense_k_assembled[..., :half_d]

                                    dense_k_rot = workspace_k_rot[:, :, :L_dense]
                                    # Compute RoPE in-place in the pre-allocated slice
                                    torch.mul(dense_k_assembled[..., :rotary_dim], cos_dense, out=dense_k_rot[..., :rotary_dim])
                                    dense_k_rot[..., :rotary_dim].addcmul_(dense_k_half, sin_dense)
                                    if rotary_dim < head_dim:
                                        dense_k_rot[..., rotary_dim:].copy_(dense_k_assembled[..., rotary_dim:])
                                else:
                                    dense_k_rot = None
                                if not has_dense and not has_comp:
                                    attn_out_b = torch.zeros((1, H_q, 1, D), dtype=query_states.dtype, device=query_states.device)
                                elif has_dense and not has_comp:
                                    # Dense window only: use standard SDPA
                                    k_rep = repeat_kv(dense_k_rot, num_key_value_groups)
                                    # Slice V to L_dense (actual valid tokens in MPS path) for SDPA
                                    v_rep = repeat_kv(dense_v_assembled[:, :, :L_dense], num_key_value_groups)
                                    attn_out_b = F.scaled_dot_product_attention(
                                        query_states[b_idx:b_idx+1], k_rep, v_rep,
                                        is_causal=False,
                                    )
                                elif has_comp and not has_dense:
                                    # Compressed history only.
                                    # Phase 2: Custom Metal compute shader path (macOS).
                                    # Falls back to Phase 1 C++ ATen or Python.
                                    _Q_sq = query_states[b_idx, :, 0, :]  # [H_q, D]
                                    _ca = _cos_anc_2d if _cos_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    _sa = _sin_anc_2d if _sin_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    # ATen kernel derives rotary_dim from tensor shape (not a
                                    # separate param) -- must get the true unpadded width, not
                                    # the Metal-oriented padded buffer above.
                                    _ca_true = _cos_anc_2d_true if _cos_anc_2d_true is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    _sa_true = _sin_anc_2d_true if _sin_anc_2d_true is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)

                                    has_residual = False
                                    if pool is not None and getattr(pool, "residual_K_positions", None) is not None:
                                        # B1: read the cached flag (set at write_block time) instead of
                                        # calling .item() on a device tensor every layer every step.
                                        has_residual = getattr(pool, "has_any_residual", False)

                                    _res_pos_K = pool.residual_K_positions if pool.residual_K_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _res_val_K = pool.residual_K_values if pool.residual_K_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _res_pos_V = pool.residual_V_positions if pool.residual_V_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _res_val_V = pool.residual_V_values if pool.residual_V_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _fact_pos = pool.fact_anchor_positions if pool.fact_anchor_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _fact_val_K = pool.fact_anchors_K if pool.fact_anchors_K is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _fact_val_V = pool.fact_anchors_V if pool.fact_anchors_V is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)

                                    if _DKV_HAS_METAL_ATTN and pool is not None:
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        # Binding grew 4 trailing dense-window args (dense_K/V +
                                        # cos/sin_dense); this caller merges dense separately, so
                                        # pass empties (numel==0 → impl skips the dense loop).
                                        _ed = torch.empty(0, device=_Q_sq.device, dtype=_Q_sq.dtype)
                                        out_sparse, _ = _decode_attention_metal(
                                            _Q_sq.contiguous(),
                                            pool.U.contiguous(),
                                            pool.U_scale.contiguous(),
                                            pool.V_K.contiguous(),
                                            pool.V_V.contiguous(),
                                            pool.anchors_K.contiguous(),
                                            pool.anchors_V.contiguous(),
                                            pool.seq_lens.contiguous(),
                                            pool.scales.contiguous(),
                                            _ca,
                                            _sa,
                                            block_indices.contiguous(),
                                            _scale,
                                            num_heads,
                                            num_key_value_heads,
                                            kv_manager.rank,
                                            _res_pos_K.contiguous(),
                                            _res_val_K.contiguous(),
                                            _res_pos_V.contiguous(),
                                            _res_val_V.contiguous(),
                                            _fact_pos.contiguous(),
                                            _fact_val_K.contiguous(),
                                            _fact_val_V.contiguous(),
                                            _ed, _ed, _ed, _ed,
                                            _anchor_rotary_dim,
                                        )
                                    elif _DKV_HAS_DECODE_ATTN and pool is not None and not has_residual:
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        out_sparse = _decode_attention_aten(
                                            _Q_sq.contiguous(),
                                            pool.U.contiguous(),
                                            pool.U_scale.contiguous(),
                                            pool.V_K.contiguous(),
                                            pool.V_V.contiguous(),
                                            pool.anchors_K.contiguous(),
                                            pool.anchors_V.contiguous(),
                                            pool.seq_lens.contiguous(),
                                            pool.scales.contiguous(),
                                            _ca_true,
                                            _sa_true,
                                            block_indices.contiguous(),
                                            _scale,
                                            num_heads,
                                            num_key_value_heads,
                                            kv_manager.rank,
                                        )  # [H_q, D] float16
                                    else:
                                        # Python fallback
                                        _bsizes = pool.seq_lens[block_indices]
                                        out_sparse, _ = fused_decode_mps(
                                            Q                    = _Q_sq,
                                            pool                 = pool,
                                            block_indices        = block_indices,
                                            blk_sizes            = _bsizes,
                                            num_key_value_groups = num_key_value_groups,
                                            anchor_indices       = anchor_indices,
                                            cos                  = cos_all,
                                            sin                  = sin_all,
                                        )
                                    attn_out_b = out_sparse.unsqueeze(0).unsqueeze(2)  # [1, H_q, 1, D]
                                else:
                                    # Both present: run independently and combine via LSE
                                    # 1. Dense SDPA path
                                    k_rep = repeat_kv(dense_k_rot, num_key_value_groups)
                                    v_rep = repeat_kv(dense_v_assembled[:, :, :L_dense], num_key_value_groups)
                                    out_dense = F.scaled_dot_product_attention(
                                        query_states[b_idx:b_idx+1], k_rep, v_rep,
                                        is_causal=False,
                                    )  # [1, H_q, 1, D]
                                    out_dense_hd = out_dense[0, :, 0, :].float()

                                    # 2. LSE for dense scores (in fp16 to avoid large fp32 promotions)
                                    _q = query_states[b_idx, :, 0, :]
                                    _kd = k_rep[0]
                                    _scale = (D ** -0.5)
                                    scores_dense = torch.matmul(_kd, _q.unsqueeze(-1)).squeeze(-1) * _scale  # [H_q, T]
                                    lse_dense = torch.logsumexp(scores_dense.float(), dim=-1)  # [H_q]

                                    # 3. Compressed history path — with LSE for combination.
                                    # Phase 2: Custom Metal compute shader path (macOS).
                                    # Falls back to Phase 1 C++ ATen or Python.
                                    _Q_sq = query_states[b_idx, :, 0, :]
                                    _ca = _cos_anc_2d if _cos_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    _sa = _sin_anc_2d if _sin_anc_2d is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    # ATen kernel derives rotary_dim from tensor shape (not a
                                    # separate param) -- must get the true unpadded width, not
                                    # the Metal-oriented padded buffer above.
                                    _ca_true = _cos_anc_2d_true if _cos_anc_2d_true is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)
                                    _sa_true = _sin_anc_2d_true if _sin_anc_2d_true is not None else torch.empty(0, device=query_states.device, dtype=torch.float32)

                                    has_residual = False
                                    if pool is not None and getattr(pool, "residual_K_positions", None) is not None:
                                        # B1: same cached flag used in both branches.
                                        has_residual = getattr(pool, "has_any_residual", False)

                                    _res_pos_K = pool.residual_K_positions if pool.residual_K_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _res_val_K = pool.residual_K_values if pool.residual_K_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _res_pos_V = pool.residual_V_positions if pool.residual_V_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _res_val_V = pool.residual_V_values if pool.residual_V_values is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _fact_pos = pool.fact_anchor_positions if pool.fact_anchor_positions is not None else torch.empty(0, device=query_states.device, dtype=torch.int16)
                                    _fact_val_K = pool.fact_anchors_K if pool.fact_anchors_K is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _fact_val_V = pool.fact_anchors_V if pool.fact_anchors_V is not None else torch.empty(0, device=query_states.device, dtype=query_states.dtype)

                                    if _DKV_HAS_METAL_ATTN and pool is not None:
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        # Same 4 trailing dense-window args as above: dense is
                                        # merged separately here, pass empties to skip it.
                                        _ed = torch.empty(0, device=_Q_sq.device, dtype=_Q_sq.dtype)
                                        out_sparse, lse_sparse = _decode_attention_metal(
                                            _Q_sq.contiguous(),
                                            pool.U.contiguous(),
                                            pool.U_scale.contiguous(),
                                            pool.V_K.contiguous(),
                                            pool.V_V.contiguous(),
                                            pool.anchors_K.contiguous(),
                                            pool.anchors_V.contiguous(),
                                            pool.seq_lens.contiguous(),
                                            pool.scales.contiguous(),
                                            _ca,
                                            _sa,
                                            block_indices.contiguous(),
                                            _scale,
                                            num_heads,
                                            num_key_value_heads,
                                            kv_manager.rank,
                                            _res_pos_K.contiguous(),
                                            _res_val_K.contiguous(),
                                            _res_pos_V.contiguous(),
                                            _res_val_V.contiguous(),
                                            _fact_pos.contiguous(),
                                            _fact_val_K.contiguous(),
                                            _fact_val_V.contiguous(),
                                            _ed, _ed, _ed, _ed,
                                            _anchor_rotary_dim,
                                        )
                                    elif _DKV_HAS_DECODE_ATTN and pool is not None and not has_residual:
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        out_sparse, lse_sparse = _decode_attention_aten_lse(
                                            _Q_sq.contiguous(),
                                            pool.U.contiguous(),
                                            pool.U_scale.contiguous(),
                                            pool.V_K.contiguous(),
                                            pool.V_V.contiguous(),
                                            pool.anchors_K.contiguous(),
                                            pool.anchors_V.contiguous(),
                                            pool.seq_lens.contiguous(),
                                            pool.scales.contiguous(),
                                            _ca_true,
                                            _sa_true,
                                            block_indices.contiguous(),
                                            _scale,
                                            num_heads,
                                            num_key_value_heads,
                                            kv_manager.rank,
                                        )  # [H_q, D] float16, [H_q] float32
                                    else:
                                        # Python fallback
                                        _bsizes = pool.seq_lens[block_indices]
                                        out_sparse, lse_sparse = fused_decode_mps(
                                            Q                    = _Q_sq,
                                            pool                 = pool,
                                            block_indices        = block_indices,
                                            blk_sizes            = _bsizes,
                                            num_key_value_groups = num_key_value_groups,
                                            anchor_indices       = anchor_indices,
                                            cos                  = cos_all,
                                            sin                  = sin_all,
                                        )  # [H_q, D], [H_q]

                                    # 4. Combine outputs via LSE safely (unclamped to preserve true log-sum-exp combination math)
                                    # NaN guards (not just Inf, MLX parity) — see the 3-way combine above
                                    # for why torch.maximum makes this necessary, not optional.
                                    lse_sparse = _apply_sparse_bias(lse_sparse, lse_dense)
                                    lse_max = torch.maximum(lse_dense, lse_sparse)
                                    lse_max_masked = lse_max.clone()
                                    lse_max_masked[~torch.isfinite(lse_max)] = 0.0

                                    w_dense = torch.exp(lse_dense - lse_max_masked)
                                    w_sparse = torch.exp(lse_sparse - lse_max_masked)

                                    w_dense[~torch.isfinite(lse_dense)] = 0.0
                                    w_sparse[~torch.isfinite(lse_sparse)] = 0.0

                                    denom = w_dense + w_sparse
                                    denom_safe = torch.clamp(denom, min=1e-9)

                                    out_sparse_fp32 = out_sparse.float()
                                    out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                                                 out_sparse_fp32 * w_sparse.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
                                    # Final safety net (MLX parity): zero any NaN that still made it
                                    # through (e.g. from a NaN in an attention output itself).
                                    out_final = torch.nan_to_num(out_final, nan=0.0)
                                    attn_out_b = out_final.to(query_states.dtype).unsqueeze(0).unsqueeze(2)  # [1, H_q, 1, D]
                        else:
                            # ── CUDA: use fused combined kernel (single dispatch for
                            # compressed blocks + dense window) when Triton is available.
                            # Falls back to native_triton_sparse_attn_decode (which does its
                            # own inline dense LSE-merge) on non-CUDA or on kernel error.
                            _use_combined = (
                                HAS_TRITON
                                and query_states.device.type == "cuda"
                                and pool is not None
                                and block_indices is not None
                                and block_indices.numel() > 0
                                # Deliberately still a RAW STRING test, not
                                # resolve_sparse_bias(). 'auto' now resolves to a
                                # 0.0 bias, but letting it satisfy this gate too
                                # would ALSO switch production from the
                                # non-combined kernel to the combined one in the
                                # same change -- two variables moving at once, and
                                # no way to attribute a recall shift to either.
                                # Kernel choice stays exactly as it is today.
                                and os.environ.get("DKV_SPARSE_BIAS", "0.0").strip().lower() in ("0", "0.0", "", "false", "off")
                            )
                            if _use_combined:
                                # Assemble pre-RoPE-rotated dense_k/dense_v for the combined kernel.
                                # The combined Triton kernel expects [1, H_kv, max_dense_len, D]
                                # pre-RoPE-rotated workspace (fixed shape).
                                # Use ONLY dense_k_assembled — the separate dense_blocks loop would
                                # double-count (dense_k_assembled already contains anchor + active data
                                # from every dense block, assembled by assemble_dense_window_kv).
                                _dk_combined = None
                                _dv_combined = None
                                if dense_k_assembled is not None and dense_len > 0:
                                    _max_dense = dense_k_assembled.shape[2]  # = max_dense_len (fixed)
                                    # Build a fixed-size [max_dense_len] position tensor for RoPE.
                                    # Padding slots (dense_len..max_dense_len-1) map to position 0;
                                    # their rotated zeros are masked in the kernel via L_dense_valid.
                                    # OPT (P1-7): cache this tensor across decode steps. The dense-window
                                    # layout is stable between steps (only changes on routing change or
                                    # block growth). routing_version already invalidates decode_cos_sliced
                                    # on any routing change — reuse the same event here.
                                    _dense_anchor_sig = tuple(
                                        int(_blk.anchor_idx) for _blk in (dense_blocks or [])
                                    )
                                    _dense_lengths = tuple(
                                        len(_blk.token_indices) for _blk in (dense_blocks or [])
                                    )
                                    # dense_len changes on every decode token.  It is
                                    # not a valid cache key: the position tensor has a
                                    # fixed max_dense_len shape and only its appended
                                    # suffix needs updating.
                                    _cache_key = (current_version, _dense_anchor_sig, _max_dense)
                                    _dpc_dict = session_dict.setdefault("cuda_dense_pos_tensor_cache", {})
                                    _dp2_cache = _dpc_dict.get(captured_layer_idx)
                                    if (_dp2_cache is not None
                                            and _dp2_cache[0] == _cache_key
                                            and _dp2_cache[2].shape[0] == _max_dense):
                                        _old_lengths = _dp2_cache[1]
                                        _dp2 = _dp2_cache[2]
                                        _cos_d2 = _dp2_cache[3]
                                        _sin_d2 = _dp2_cache[4]
                                        _offset = 0
                                        for _old_len, _new_len, _blk in zip(
                                            _old_lengths, _dense_lengths, dense_blocks or []
                                        ):
                                            if _new_len > _old_len:
                                                _s = _offset + _old_len
                                                _e = _offset + _new_len
                                                _dp2[_s:_e].copy_(torch.tensor(
                                                    _blk.token_indices[_old_len:_new_len],
                                                    dtype=torch.long,
                                                    device=_dp2.device,
                                                ))
                                                # reshape by the table's OWN last dim, not
                                                # head_dim. cos_all is [.., seq, rotary_dim]
                                                # (64 on Qwen3.5, vs head_dim 256), so
                                                # reshape(-1, head_dim) silently repacked FOUR
                                                # positions into each row and then raised on the
                                                # copy below (64 vs 256). reshape does not
                                                # validate semantics, only element count, so this
                                                # was invisible until a model had
                                                # rotary_dim != head_dim.
                                                _rope_w = cos_all.shape[-1]
                                                _cos_flat = cos_all.reshape(-1, _rope_w)
                                                _sin_flat = sin_all.reshape(-1, _rope_w)
                                                _seq_lim = _cos_flat.shape[0]
                                                _cos_d2[:, :, _s:_e].copy_(
                                                    _cos_flat[_dp2[_s:_e].clamp(min=0, max=_seq_lim - 1)].unsqueeze(0).unsqueeze(1)
                                                )
                                                _sin_d2[:, :, _s:_e].copy_(
                                                    _sin_flat[_dp2[_s:_e].clamp(min=0, max=_seq_lim - 1)].unsqueeze(0).unsqueeze(1)
                                                )
                                            _offset += _new_len
                                        _dpc_dict[captured_layer_idx] = (
                                            _cache_key, _dense_lengths, _dp2, _cos_d2, _sin_d2
                                        )
                                    else:
                                        _dp2 = torch.zeros(_max_dense, dtype=torch.long, device=query_states.device)
                                        # Concatenate on the HOST, then transfer ONCE.
                                        # This loop used to build a device tensor per dense
                                        # block -- torch.tensor(python_list, device=cuda) is
                                        # a pageable host->device copy, i.e. a sync, and the
                                        # sync recorder ranked this line #1 at 30,184 of
                                        # 61,338 recorded syncs (49%). That is 5.5 per layer
                                        # per token, which is exactly L_dense/block_size =
                                        # 1538/256 = 6 dense blocks. One transfer instead.
                                        # Order and truncation are unchanged: the old loop
                                        # appended token_indices block by block and stopped
                                        # at _max_dense, which is this concat-then-truncate.
                                        _flat2 = []
                                        for _blk in (dense_blocks or []):
                                            if _blk.token_indices:
                                                _flat2.extend(_blk.token_indices)
                                                if len(_flat2) >= _max_dense:
                                                    break
                                        del _flat2[_max_dense:]
                                        _pos2 = len(_flat2)
                                        if _pos2:
                                            _dp2[:_pos2].copy_(
                                                torch.tensor(_flat2, dtype=torch.long, device=_dp2.device)
                                            )

                                        _cos_d2 = cos_all[0, _dp2.clamp(min=0, max=cos_all.shape[1] - 1)]
                                        _sin_d2 = sin_all[0, _dp2.clamp(min=0, max=sin_all.shape[1] - 1)]
                                        if _cos_d2.dim() == 3:
                                            _cos_d2 = _cos_d2.squeeze(1)
                                            _sin_d2 = _sin_d2.squeeze(1)
                                        _cos_d2 = _cos_d2.unsqueeze(0).unsqueeze(1)  # [1,1,max_dense_len,D]
                                        _sin_d2 = _sin_d2.unsqueeze(0).unsqueeze(1)
                                        # Store per-layer (matches the cache-hit branch above,
                                        # which writes _dpc_dict[captured_layer_idx]).  Writing a
                                        # bare tuple to session_dict here would clobber the dict
                                        # created by setdefault(...,{}) at the top of this block,
                                        # so the next layer's .get() would crash on a tuple.
                                        _dpc_dict[captured_layer_idx] = (
                                            _cache_key, _dense_lengths, _dp2, _cos_d2, _sin_d2
                                        )
                                    # Keep the dense KV in its RoPE-rotated form across
                                    # decode steps.  The dense window changes by only one
                                    # token at a time, but the old code rotated all
                                    # max_dense_len tokens for every layer/token.  That
                                    # made the CUDA path redo roughly 1,419 * 48 RoPE
                                    # operations per generated token, while HF's dense
                                    # cache rotates each key once when it is appended.
                                    #
                                    # Rebuild only when routing changes or a dense block is
                                    # removed.  Otherwise update the appended suffix of
                                    # each block in-place and reuse the existing rotation.
                                    _rot_cache = session_dict.setdefault("dense_rot_state", {})
                                    _rot_state = _rot_cache.get(captured_layer_idx)
                                    _layout_sig = tuple(
                                        (int(_blk.anchor_idx), len(_blk.token_indices))
                                        for _blk in (dense_blocks or [])
                                    )
                                    _lengths = tuple(item[1] for item in _layout_sig)
                                    _anchors = tuple(item[0] for item in _layout_sig)
                                    # Partial-RoPE geometry, shared by the full-rebuild
                                    # and incremental-append branches below. half_r must
                                    # be rotary_dim/2, never head_dim/2 -- see the note in
                                    # the rebuild branch.
                                    _rot_dim = _cos_d2.shape[-1]
                                    _half_r = _rot_dim // 2
                                    _head_dim_full = dense_k_assembled.shape[-1]
                                    _rot_valid = (
                                        _rot_state is not None
                                        and _rot_state.get("version") == current_version
                                        and _rot_state.get("anchors") == _anchors
                                        and len(_rot_state.get("lengths", ())) == len(_lengths)
                                        and all(
                                            new_len >= old_len
                                            for old_len, new_len in zip(
                                                _rot_state.get("lengths", ()), _lengths
                                            )
                                        )
                                        and _rot_state["rot"].shape == dense_k_assembled.shape
                                    )

                                    if not _rot_valid:
                                        # Partial RoPE (Qwen3.5-style partial_rotary_factor<1.0).
                                        #
                                        # This branch used to rotate the FULL head_dim: it paired
                                        # dim d with d +/- head_dim/2 and multiplied the whole
                                        # tensor by cos. On Qwen3.5 (head_dim 256, rotary_dim 64)
                                        # that raised outright --
                                        #   "size of tensor a (256) must match tensor b (64)" --
                                        # because cos/sin are only rotary_dim wide. The MPS branch
                                        # a few hundred lines up already did this correctly; this
                                        # CUDA branch never got the port, so Qwen3.5 could not
                                        # decode on CUDA at all.
                                        #
                                        # Rotate only [0, rotary_dim), pairing within that range
                                        # (half_r = rotary_dim/2, NOT head_dim/2 -- that pulls the
                                        # partner from an unrotated dimension), and pass the tail
                                        # through unrotated. Reduces to the previous behaviour
                                        # exactly when rotary_dim == head_dim.
                                        _rot_dim = _cos_d2.shape[-1]
                                        _half_r = _rot_dim // 2
                                        _cos_compute = _cos_d2.to(dense_k_assembled.dtype)
                                        _sin_compute = _sin_d2.to(dense_k_assembled.dtype)
                                        _dk_half2 = torch.empty_like(
                                            dense_k_assembled[..., :_rot_dim])
                                        _dk_half2[..., :_half_r] = -dense_k_assembled[..., _half_r:_rot_dim]
                                        _dk_half2[..., _half_r:] = dense_k_assembled[..., :_half_r]
                                        _rot = torch.empty_like(dense_k_assembled)
                                        torch.mul(dense_k_assembled[..., :_rot_dim], _cos_compute,
                                                  out=_rot[..., :_rot_dim])
                                        _rot[..., :_rot_dim].addcmul_(_dk_half2, _sin_compute)
                                        if _rot_dim < dense_k_assembled.shape[-1]:
                                            _rot[..., _rot_dim:].copy_(
                                                dense_k_assembled[..., _rot_dim:])
                                        _rot_state = {
                                            "version": current_version,
                                            "anchors": _anchors,
                                            "lengths": _lengths,
                                            "rot": _rot,
                                            "half": _dk_half2,
                                            "cos": _cos_compute,
                                            "sin": _sin_compute,
                                        }
                                        _rot_cache[captured_layer_idx] = _rot_state
                                    else:
                                        _rot = _rot_state["rot"]
                                        _dk_half2 = _rot_state["half"]
                                        _cos_compute = _rot_state["cos"]
                                        _sin_compute = _rot_state["sin"]
                                        _offset = 0
                                        for _old_len, _new_len in zip(
                                            _rot_state["lengths"], _lengths
                                        ):
                                            if _new_len > _old_len:
                                                _s = _offset + _old_len
                                                _e = _offset + _new_len
                                                _raw_suffix = dense_k_assembled[:, :, _s:_e]
                                                # Same partial-RoPE correction as the rebuild
                                                # branch: _dk_half2 is rotary_dim wide and the
                                                # pairing is within [0, rotary_dim).
                                                _half_suffix = _dk_half2[:, :, _s:_e]
                                                _half_suffix[..., :_half_r] = -_raw_suffix[..., _half_r:_rot_dim]
                                                _half_suffix[..., _half_r:] = _raw_suffix[..., :_half_r]
                                                # Only the newly appended positions
                                                # need dtype conversion; the cached
                                                # prefix is already in compute dtype.
                                                _cos_compute[:, :, _s:_e].copy_(
                                                    _cos_d2[:, :, _s:_e].to(_cos_compute.dtype)
                                                )
                                                _sin_compute[:, :, _s:_e].copy_(
                                                    _sin_d2[:, :, _s:_e].to(_sin_compute.dtype)
                                                )
                                                torch.mul(
                                                    _raw_suffix[..., :_rot_dim],
                                                    _cos_compute[:, :, _s:_e],
                                                    out=_rot[:, :, _s:_e, :_rot_dim],
                                                )
                                                _rot[:, :, _s:_e, :_rot_dim].addcmul_(
                                                    _half_suffix,
                                                    _sin_compute[:, :, _s:_e],
                                                )
                                                if _rot_dim < _head_dim_full:
                                                    _rot[:, :, _s:_e, _rot_dim:].copy_(
                                                        _raw_suffix[..., _rot_dim:]
                                                    )
                                            _offset += _new_len
                                        _rot_state["lengths"] = _lengths

                                    _dk_combined = _rot
                                    _dv_combined = dense_v_assembled
                                # P1-6: Deferred batch dispatch — queue this session's call
                                # so we can dispatch all sessions in tight Python-free sequence.
                                if _triton_batch_enabled:
                                    from native_core.kv_runtime_manager import get_layer_rank as _get_layer_rank
                                    _cfg = getattr(kv_manager, "config", None)
                                    _layer_active_rank = _get_layer_rank(
                                        captured_layer_idx, kv_manager.num_layers, kv_manager.rank,
                                        early_boost=getattr(_cfg, "early_layer_rank_boost", False),
                                        max_rank_early=getattr(_cfg, "max_rank_early", 0),
                                    )
                                    _triton_batch_queue.append((
                                        b_idx,
                                        dict(
                                            q=query_states[b_idx:b_idx+1],
                                            block_indices=block_indices,
                                            pool=pool,
                                            dense_k=_dk_combined,
                                            dense_v=_dv_combined,
                                            num_key_value_groups=num_key_value_groups,
                                            R=_layer_active_rank,
                                            S_MAX=session_mbs,
                                            anchor_indices=anchor_indices,
                                            cos=cos_all,
                                            sin=sin_all,
                                            dense_len=dense_len,
                                        ),
                                    ))
                                    # Placeholder; filled after the deferred dispatch below
                                    attn_outputs.append(None)
                                    continue  # skip the attn_outputs.append(attn_out_b) below
                                else:
                                    # Gather-cache key: (layer, streaming metadata version).
                                    # metadata_version bumps exactly when a block is
                                    # flushed into the pool, which is the only event that
                                    # changes the query-independent gather. So the cached
                                    # gather is reused for every token until the next flush.
                                    _gc = _gk = None
                                    if _DECODE_CACHE_CUDA:
                                        _smgr = getattr(kv_manager, "_streaming_mgr", None)
                                        _mdver = (_smgr._metadata_versions.get(sid, {}).get(captured_layer_idx, 0)
                                                  if _smgr is not None else 0)
                                        _gc = kv_manager.decode_workspace.setdefault(sid, {}).setdefault(
                                            "_gather_cache_cuda", {})
                                        _gk = (captured_layer_idx, _mdver)
                                    from native_core.kv_runtime_manager import get_layer_rank as _get_layer_rank
                                    _cfg = getattr(kv_manager, "config", None)
                                    _layer_active_rank = _get_layer_rank(
                                        captured_layer_idx, kv_manager.num_layers, kv_manager.rank,
                                        early_boost=getattr(_cfg, "early_layer_rank_boost", False),
                                        max_rank_early=getattr(_cfg, "max_rank_early", 0),
                                    )
                                    # ── Re-materialisation cache (DKV_REMAT_CACHE) ──
                                    # Skips the per-token U@V rebuild entirely by
                                    # materialising routed blocks once per interval and
                                    # attending the result with a plain SDPA. See
                                    # native_core/sparse_decode/remat_cache.py.
                                    _remat_out = None
                                    if _REMAT_ENABLED and block_indices is not None \
                                            and block_indices.numel() > 0:
                                        from native_core.sparse_decode.remat_cache import (
                                            RematCache as _RC, reconstruct_blocks as _rb,
                                            attend_with_remat as _awr,
                                        )
                                        # Lives in triton_fused_decode, not this module.
                                        from native_core.sparse_decode.triton_fused_decode import (
                                            _gather_routed_blocks_for_kernel as _grb,
                                        )
                                        _ws = kv_manager.decode_workspace.setdefault(sid, {})
                                        _rc = _ws.get("_remat_cache")
                                        if _rc is None:
                                            _rc = _RC()
                                            _ws["_remat_cache"] = _rc
                                        # Self-populate the last-DKV-layer index: the
                                        # step counter must advance ONCE PER TOKEN, not
                                        # once per layer. Without this it stays at -1,
                                        # the counter never advances, and the interval
                                        # refresh is silently dead (the cache would then
                                        # only refresh on routing/pool change).
                                        _mid = id(kv_manager)
                                        if captured_layer_idx > _LAST_DKV_LAYER.get(_mid, -1):
                                            _LAST_DKV_LAYER[_mid] = captured_layer_idx
                                        _step = _ws.get("_remat_step", 0)
                                        _smgr2 = getattr(kv_manager, "_streaming_mgr", None)
                                        _poolgen = (_smgr2._metadata_versions.get(sid, {})
                                                    .get(captured_layer_idx, 0)
                                                    if _smgr2 is not None else 0)
                                        _rkey = _RC.make_key(captured_layer_idx,
                                                             current_version, _poolgen, _step)
                                        _hit = _rc.get(_rkey)
                                        if _hit is None:
                                            _g = _grb(pool, block_indices,
                                                      anchor_indices, cos_all, sin_all)
                                            # The residuals are a CORRECTNESS
                                            # requirement, not a refinement: they
                                            # carry the exact values of the tokens
                                            # the SVD reconstructs worst (codes,
                                            # digits). Dropping them attends every
                                            # routed block at pure low-rank
                                            # fidelity and loses exact recall
                                            # while still reading fluently.
                                            # res_k is already rotated at the
                                            # residuals' TRUE token positions by
                                            # the gather (DKV_RESIDUAL_EXACT_ROPE).
                                            _has_res = _g.get("has_res", False)
                                            _Km, _Vm = _rb(
                                                _g["U"], _g["V_K"], _g["V_V"],
                                                _g["anchors_K"], _g["anchors_V"],
                                                _g["scales"], _g["U_scale"],
                                                _layer_active_rank,
                                                res_k=_g["res_k"] if _has_res else None,
                                                res_pos=_g["res_pos"] if _has_res else None,
                                                res_v=_g["res_v"] if _has_res else None,
                                                res_pos_v=_g["res_pos_v"] if _has_res else None)
                                            _rc.put(_rkey, _Km, _Vm)
                                            # clone: under DKV_STATIC_GATHER the gather
                                            # returns a PERSISTENT buffer that the next
                                            # layer's gather overwrites, and this is held
                                            # across tokens. Same invariant the gather's
                                            # batch-queue note states — anything outliving
                                            # the call must own its memory. _Km/_Vm are
                                            # fresh bmm outputs, so only this needs it.
                                            _seq_cached = _g["seq_lens"].clone()
                                            _ws[("_remat_seq", captured_layer_idx)] = _seq_cached
                                        else:
                                            _Km, _Vm = _hit
                                            _seq_cached = _ws[("_remat_seq", captured_layer_idx)]
                                        _remat_out = _awr(
                                            query_states[b_idx:b_idx+1], _Km, _Vm,
                                            _seq_cached, _dk_combined, _dv_combined,
                                            dense_len, num_key_value_groups)
                                        # Advance once per token, on the LAST DKV layer.
                                        # `_LAST_DKV_LAYER` learns the max layer index by
                                        # watching layers go by, so on the FIRST token it
                                        # equals the current layer at every layer and the
                                        # counter advanced once per LAYER (~28x) instead of
                                        # once per token. Harmless for output but it shifts
                                        # every interval boundary, which would confound an
                                        # interval sweep. Only advance once the highest
                                        # layer index has actually stopped growing, i.e.
                                        # from the second token onward.
                                        _last = _LAST_DKV_LAYER.get(_mid, -1)
                                        if captured_layer_idx == _last and _ws.get("_remat_saw_last"):
                                            _ws["_remat_step"] = _step + 1
                                        if captured_layer_idx == _last:
                                            _ws["_remat_saw_last"] = True

                                    attn_out_b = _remat_out if _remat_out is not None else \
                                        native_triton_sparse_attn_decode_combined(
                                        q=query_states[b_idx:b_idx+1],
                                        block_indices=block_indices,
                                        pool=pool,
                                        dense_k=_dk_combined,
                                        dense_v=_dv_combined,
                                        num_key_value_groups=num_key_value_groups,
                                        R=_layer_active_rank,
                                        S_MAX=session_mbs,
                                        anchor_indices=anchor_indices,
                                        cos=cos_all,
                                        sin=sin_all,
                                        dense_len=dense_len,
                                        gather_cache=_gc,
                                        gather_key=_gk,
                                    )
                            else:
                                from native_core.kv_runtime_manager import get_layer_rank as _get_layer_rank
                                _cfg = getattr(kv_manager, "config", None)
                                _layer_active_rank = _get_layer_rank(
                                    captured_layer_idx, kv_manager.num_layers, kv_manager.rank,
                                    early_boost=getattr(_cfg, "early_layer_rank_boost", False),
                                    max_rank_early=getattr(_cfg, "max_rank_early", 0),
                                )
                                attn_out_b = native_triton_sparse_attn_decode(
                                    q=query_states[b_idx:b_idx+1],
                                    block_indices=block_indices,
                                    pool=pool,
                                    dense_blocks=dense_blocks,
                                    active_k=dense_k_assembled,
                                    active_v=dense_v_assembled,
                                    num_key_value_groups=num_key_value_groups,
                                    R=_layer_active_rank,
                                    S_MAX=session_mbs,
                                    anchor_indices=anchor_indices,
                                    cos=cos_all,
                                    sin=sin_all,
                                    total_seq_len=total_seq_len,
                                    max_valid_len=max_valid_len,
                                    cos_sliced=cos_sliced_arg,
                                    sin_sliced=sin_sliced_arg,
                                    session_id=sid,
                                    layer_idx=captured_layer_idx,
                                    decode_workspace=kv_manager.decode_workspace,
                                    active_len=dense_len,
                                )

                        # ── Validation: compare SRL output vs. full-attention output ──
                        if _validate_this_step:
                            try:
                                _full_bi, _full_dn, _full_ai, _full_mai, _full_mvl = kv_manager.get_cached_decode_blocks(
                                    sid, captured_layer_idx, query_states.device
                                )
                                with torch.no_grad():
                                    if _is_mps_decode:
                                        has_dense = (dense_k_assembled is not None and dense_len > 0)
                                        if has_dense:
                                            # Validation path: slice to dense_len for RoPE (shape-safe)
                                            dense_k_valid = dense_k_assembled[:, :, :dense_len]
                                            dense_positions_list = []
                                            for blk in dense_blocks:
                                                dense_positions_list.extend(blk.token_indices)
                                            dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                            # Same rotary_dim-vs-head_dim reshape hazard as
                                            # the cached dense-position path above.
                                            _rope_w = cos_all.shape[-1]
                                            cos_flat = cos_all.reshape(-1, _rope_w)
                                            sin_flat = sin_all.reshape(-1, _rope_w)
                                            seq_limit = cos_flat.shape[0]
                                            cos_dense = cos_flat[dense_positions.clamp(min=0, max=seq_limit - 1)].unsqueeze(0).unsqueeze(1)
                                            sin_dense = sin_flat[dense_positions.clamp(min=0, max=seq_limit - 1)].unsqueeze(0).unsqueeze(1)
                                            dense_k_rot = _apply_rope_single(dense_k_valid, cos_dense, sin_dense)

                                        _q_val = query_states[b_idx, :, 0, :]
                                        _full_bsizes = pool.seq_lens[_full_bi]
                                        out_sparse_full, lse_sparse_full = fused_decode_mps(
                                            Q                    = _q_val,
                                            pool                 = pool,
                                            block_indices        = _full_bi,
                                            blk_sizes            = _full_bsizes,
                                            num_key_value_groups = num_key_value_groups,
                                            anchor_indices       = _full_ai,
                                            cos                  = cos_all,
                                            sin                  = sin_all,
                                        )
                                        
                                        has_dense = (dense_k_assembled is not None and dense_len > 0)
                                        has_comp  = (_full_bi is not None and _full_bi.numel() > 0)
                                        
                                        if not has_dense and not has_comp:
                                            attn_out_full_approx = torch.zeros((1, query_states.shape[1], 1, query_states.shape[3]), dtype=query_states.dtype, device=query_states.device)
                                        elif has_dense and not has_comp:
                                            k_rep = repeat_kv(dense_k_rot, num_key_value_groups)
                                            v_rep = repeat_kv(dense_v_assembled[:, :, :dense_len], num_key_value_groups)
                                            attn_out_full_approx = F.scaled_dot_product_attention(
                                                query_states[b_idx:b_idx+1], k_rep, v_rep,
                                                is_causal=False,
                                            )
                                        elif has_comp and not has_dense:
                                            attn_out_full_approx = out_sparse_full.unsqueeze(0).unsqueeze(2)
                                        else:
                                            k_rep = repeat_kv(dense_k_rot, num_key_value_groups)
                                            v_rep = repeat_kv(dense_v_assembled[:, :, :dense_len], num_key_value_groups)
                                            out_dense_val = F.scaled_dot_product_attention(
                                                query_states[b_idx:b_idx+1], k_rep, v_rep,
                                                is_causal=False,
                                            )
                                            
                                            _kd = k_rep[0]
                                            _scale = (query_states.shape[3] ** -0.5)
                                            scores_dense = torch.matmul(_kd, _q_val.unsqueeze(-1)).squeeze(-1) * _scale
                                            lse_dense_val = torch.logsumexp(scores_dense.float(), dim=-1)
                                            
                                            out_dense_hd = out_dense_val[0, :, 0, :].float()
                                            out_sparse_full_fp32 = out_sparse_full.float()
                                            lse_dense_fp32 = lse_dense_val.to(torch.float32)
                                            lse_sparse_full_fp32 = lse_sparse_full.to(torch.float32)

                                            lse_max_full = torch.maximum(lse_dense_fp32, lse_sparse_full_fp32)
                                            lse_max_full_masked = lse_max_full.clone()
                                            lse_max_full_masked[torch.isinf(lse_max_full)] = 0.0

                                            w_dense_full = torch.exp(lse_dense_fp32 - lse_max_full_masked)
                                            w_sparse_full = torch.exp(lse_sparse_full_fp32 - lse_max_full_masked)

                                            w_dense_full[torch.isinf(lse_dense_fp32)] = 0.0
                                            w_sparse_full[torch.isinf(lse_sparse_full_fp32)] = 0.0

                                            denom_full = w_dense_full + w_sparse_full
                                            denom_full_safe = torch.clamp(denom_full, min=1e-9)
                                            out_final_full = (out_dense_hd * w_dense_full.unsqueeze(-1) +
                                                              out_sparse_full_fp32 * w_sparse_full.unsqueeze(-1)) / denom_full_safe.unsqueeze(-1)
                                            attn_out_full_approx = out_final_full.to(query_states.dtype).unsqueeze(0).unsqueeze(2)
                                    else:
                                        from native_core.kv_runtime_manager import get_layer_rank as _get_layer_rank
                                        _cfg = getattr(kv_manager, "config", None)
                                        _layer_active_rank = _get_layer_rank(
                                            captured_layer_idx, kv_manager.num_layers, kv_manager.rank,
                                            early_boost=getattr(_cfg, "early_layer_rank_boost", False),
                                            max_rank_early=getattr(_cfg, "max_rank_early", 0),
                                        )
                                        attn_out_full_approx = native_triton_sparse_attn_decode(
                                            q=query_states[b_idx:b_idx+1],
                                            block_indices=_full_bi,
                                            pool=pool,
                                            dense_blocks=_full_dn,
                                            active_k=dense_k_assembled,
                                            active_v=dense_v_assembled,
                                            num_key_value_groups=num_key_value_groups,
                                            R=_layer_active_rank,
                                            S_MAX=session_mbs,
                                            anchor_indices=_full_ai,
                                            cos=cos_all,
                                            sin=sin_all,
                                            total_seq_len=total_seq_len,
                                            max_valid_len=_full_mvl,
                                            cos_sliced=None,
                                            sin_sliced=None,
                                        )

                                rel_err = (
                                    (attn_out_full_approx.float() - attn_out_b.float()).norm()
                                    / (attn_out_full_approx.float().norm() + 1e-8)
                                ).item()
                                n_sel = block_indices.numel() if block_indices is not None else 0
                                n_full = _full_bi.numel() if _full_bi is not None else 0
                                frac = n_sel / max(n_full, 1)
                                print(
                                    f"[SRL Validate] step={srl_state.current_step_count} "
                                    f"layer=0 rel_err={rel_err:.4f} "
                                    f"blocks={n_sel}/{n_full} ({frac*100:.1f}%)"
                                )
                            except Exception as _ve:
                                import traceback
                                print(f"[SRL Validate] Exception during validation: {_ve}")
                                traceback.print_exc()

                        attn_outputs.append(attn_out_b)

                    # P1-6: Deferred Triton batch dispatch — fire all queued sessions
                    # in tight Python-free sequence on the default CUDA stream.
                    if _triton_batch_queue:
                        for _b_deferred, _kwargs_deferred in _triton_batch_queue:
                            _out_deferred = native_triton_sparse_attn_decode_combined(**_kwargs_deferred)
                            attn_outputs[_b_deferred] = _out_deferred

                    attn_output = torch.cat(attn_outputs, dim=0)

                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                    if attn_gate is not None:
                        attn_output = attn_output * torch.sigmoid(attn_gate)
                    attn_output = self.o_proj(attn_output)

                    # transformers 4.44-4.47: fixed 3-tuple return; hand back the
                    # cache object 4.46 passed in so the model can finalize it.
                    if _new_cache_convention:
                        return (attn_output, None)
                    return (attn_output, None, past_key_value)

                # ==============================================================
                # PREFILL / MULTI-QUERY PATH
                # ==============================================================
                if use_cache:
                    # Only a prior-turn COMPRESSED block set is compressed-history
                    # prefill.  During a fresh long prompt, the current prompt's
                    # earlier chunks are also present as ACCUMULATING blocks, but
                    # treating those as "compressed history" selects the
                    # incremental branch too early and diverges from MLX's raw
                    # prefill cache behavior.
                    has_compressed_history = False
                    for sid in session_ids:
                        if sid != "dummy_session":
                            blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                            if any(
                                getattr(b, "state", None) == "COMPRESSED"
                                and getattr(b, "U", None) is not None
                                and getattr(b, "V", None) is not None
                                for b in blocks
                            ):
                                has_compressed_history = True
                                break

                    if has_compressed_history:
                        # ── INCREMENTAL PREFILL (2nd+ turn) ─────────────────────────────────
                        # Compressed history already exists in the pool from a prior turn.
                        # Chunk the new query sequence to avoid high peak memory on MPS.
                        _chunk_sid = next(
                            (x for x in session_ids if x != "dummy_session"),
                            "default",
                        )
                        _chunk_size = _get_prefill_chunk_size(
                            kv_manager, _chunk_sid, query_states.device
                        )
                        seq_lens = []
                        for b_idx, sid in enumerate(session_ids):
                            if sid == "dummy_session":
                                seq_lens.append(q_len)
                            else:
                                seq_lens.append(kv_manager.get_session_sequence_length(sid))

                        attn_outputs = []
                        for b_idx, sid in enumerate(session_ids):
                            if sid == "dummy_session":
                                attn_outputs.append(
                                    torch.zeros((1, num_heads, q_len, head_dim),
                                                device=query_states.device, dtype=query_states.dtype)
                                )
                                continue

                            curr_q = query_states[b_idx:b_idx+1]
                            curr_k = key_states[b_idx:b_idx+1]
                            curr_v = value_states[b_idx:b_idx+1]
                            curr_unrot_k = unrot_key_states[b_idx:b_idx+1]

                            num_chunks = math.ceil(q_len / _chunk_size)
                            chunk_outs = []
                            for c in range(num_chunks):
                                c_start = c * _chunk_size
                                c_end = min((c + 1) * _chunk_size, q_len)
                                c_len = c_end - c_start

                                chunk_q = curr_q[:, :, c_start:c_end, :]
                                chunk_k = curr_k[:, :, c_start:c_end, :]
                                chunk_v = curr_v[:, :, c_start:c_end, :]
                                chunk_unrot_k = curr_unrot_k[:, :, c_start:c_end, :]

                                # 1. Path A: Causal Local Self-Attention over this chunk
                                out_local, lse_local = _flash_local_attention(chunk_q, chunk_k, chunk_v)

                                # 2. Path B: History Cross-Attention (compressed & dense blocks)
                                K_b = seq_lens[b_idx] + c_start
                                out_hist_dense, lse_hist_dense = None, None
                                out_hist_comp, lse_hist_comp   = None, None

                                if K_b > 0:
                                    blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                                    history_blocks = [b for b in blocks if b.anchor_idx < K_b]
                                    history_blocks = _sparse_prefill_filter_blocks(history_blocks, chunk_q, chunk_start=K_b)

                                    comp_blocks = []
                                    dense_k = []
                                    dense_v = []
                                    dense_positions_list = []

                                    for b in history_blocks:
                                        if getattr(b, "state", None) == "COMPRESSED" \
                                                and b.U is not None and b.V is not None:
                                            comp_blocks.append(b)
                                        else:
                                            ak = b.anchor_kv[0, 0]
                                            av = b.anchor_kv[0, 1]
                                            if b.active_k is not None:
                                                k_blk = b.active_k[0]
                                                v_blk = b.active_v[0]
                                                act_len = k_blk.shape[1]
                                            elif getattr(b, "active_k_cpu", None) is not None:
                                                k_blk = b.active_k_cpu[0].to(query_states.device, non_blocking=True)
                                                v_blk = b.active_v_cpu[0].to(query_states.device, non_blocking=True)
                                                act_len = k_blk.shape[1]
                                            else:
                                                k_blk, v_blk = None, None
                                                act_len = 0
                                            ak_u = ak.unsqueeze(1)
                                            av_u = av.unsqueeze(1)
                                            if k_blk is not None:
                                                dense_k.append(torch.cat([ak_u, k_blk], dim=1))
                                                dense_v.append(torch.cat([av_u, v_blk], dim=1))
                                            else:
                                                dense_k.append(ak_u)
                                                dense_v.append(av_u)
                                            dense_positions_list.extend(range(b.anchor_idx, b.anchor_idx + 1 + act_len))

                                    max_pos = K_b + c_len
                                    if comp_blocks:
                                        mbs = getattr(comp_blocks[0], "micro_block_size", kv_manager.get_session_micro_block_size(sid))
                                        max_pos = max(max_pos, max(b.anchor_idx for b in comp_blocks) + mbs)

                                    hist_pos = torch.arange(max_pos, device=query_states.device, dtype=torch.long).unsqueeze(0)
                                    _rot_emb_fn_inc = _resolve_rotary_emb(model)
                                    cos_all, sin_all = _rot_emb_fn_inc(value_states[b_idx:b_idx+1], hist_pos)

                                    if dense_k:
                                        k_dense = torch.cat(dense_k, dim=1).unsqueeze(0)
                                        v_dense = torch.cat(dense_v, dim=1).unsqueeze(0)

                                        dense_positions_tensor = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                        cos_dense = cos_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                        sin_dense = sin_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                        k_dense_rot = _apply_rope_single(k_dense, cos_dense, sin_dense)

                                        k_dense_rep = repeat_kv(k_dense_rot, num_key_value_groups).to(chunk_q.dtype)
                                        v_dense_rep = repeat_kv(v_dense, num_key_value_groups).to(chunk_q.dtype)

                                        _scale = 1.0 / math.sqrt(head_dim)
                                        scores_dense = torch.matmul(chunk_q * _scale, k_dense_rep.transpose(-2, -1))
                                        lse_hist_dense = torch.logsumexp(scores_dense, dim=-1)
                                        weights_dense = torch.softmax(scores_dense, dim=-1)
                                        out_hist_dense = torch.matmul(weights_dense, v_dense_rep)
                                        del scores_dense, weights_dense

                                    if comp_blocks and getattr(kv_manager, "native_pool", None) is not None:
                                        out_hist_comp, lse_hist_comp = _project_then_attend_history(
                                            chunk_q, comp_blocks, kv_manager.native_pool, sid, cos_all, sin_all
                                        )

                                # Combine Path A and Path B
                                out_chunk = _combine_outputs([
                                    (out_local,      lse_local),
                                    (out_hist_dense, lse_hist_dense),
                                    (out_hist_comp,  lse_hist_comp),
                                ])
                                chunk_outs.append(out_chunk)

                                # Capture this chunk's KV immediately
                                if sid != "dummy_session":
                                    kv_manager.capture_prefill_kv(
                                        sid, captured_layer_idx,
                                        chunk_unrot_k.detach(),
                                        chunk_v.detach(),
                                    )

                                if query_states.device.type == "mps":
                                    # Use a higher threshold to avoid frequent empty_cache slowdowns unless memory is tight
                                    _thresh = float(os.environ.get("DKV_MPS_IN_LOOP_EMPTY_CACHE_THRESHOLD_GB", "5.5")) * 1024 * 1024 * 1024
                                    if torch.mps.driver_allocated_memory() > _thresh:
                                        torch.mps.empty_cache()

                            attn_outputs.append(torch.cat(chunk_outs, dim=2))

                        attn_output = torch.cat(attn_outputs, dim=0)

                    else:
                        # ── FRESH PREFILL (1st turn / new session) ───────────────────────────
                        # Attend to previous prefill chunks of the same prompt if they exist.
                        # This is critical for progressive prompt chunking correctness!
                        _chunk_sid = next(
                            (x for x in session_ids if x != "dummy_session"),
                            "default",
                        )
                        _chunk_size = _get_prefill_chunk_size(
                            kv_manager, _chunk_sid, query_states.device
                        )
                        # B2: Pre-compute per-batch position offsets with a single .tolist()
                        # sync so the inner chunk loop reads plain Python ints, not tensors.
                        if position_ids is not None:
                            _pos_ids_cpu = position_ids[:, 0].tolist()
                        else:
                            _pos_ids_cpu = [0] * bsz
                        _global_offset = _pos_ids_cpu[0]
                        if os.environ.get("DKV_CONTIGUOUS_PREFILL", "0") == "1":
                            # ── CONTIGUOUS DENSE PREFILL (MLX-parity) — EXPERIMENTAL ─────────
                            # MLX keeps a growing ROTATED K/V buffer of all tokens so far and
                            # attends each chunk against it with ONE flash SDPA
                            # (_sparse_prefill_attend, all_k = "rotated keys, ALL tokens").
                            # The default CUDA path instead re-assembles history blocks
                            # (torch.cat), re-applies RoPE to all history, and runs an EAGER
                            # matmul+softmax+LSE-merge every chunk — the O(N^2) work behind the
                            # ~1.4x-dense forward.  This branch replicates MLX: a per-(session,
                            # layer) rotated buffer + a single SDPA with an explicit bottom-right
                            # causal mask (NOT is_causal — its non-square alignment is
                            # version-dependent; an explicit mask is unambiguous).
                            #
                            # KV is STILL captured into blocks (below) for boundary compression,
                            # so during prefill this holds the rotated buffer AND the unrotated
                            # blocks: ~2x raw prefill KV for a dense-speed forward.  The buffer
                            # is freed when decode begins (is_decode branch) and on clear_session.
                            # UNVALIDATED on GPU — A/B output_text vs the default path before use.
                            if not hasattr(kv_manager, "_contig_prefill"):
                                kv_manager._contig_prefill = {}
                            # 1x-memory variant: keep ONLY the rotated buffer, defer
                            # block creation to the compression boundary (the buffer is
                            # un-rotated there).  Saves the ~2x prefill KV by not
                            # duplicating unrotated KV into blocks each chunk.
                            _unrotate = os.environ.get("DKV_CONTIG_UNROTATE", "0") == "1"
                            if _unrotate:
                                if not hasattr(kv_manager, "_contig_chunk_lens"):
                                    kv_manager._contig_chunk_lens = {}
                                # Stash the rotary module once (used by
                                # finalize_contiguous_prefill for inverse RoPE).
                                if getattr(kv_manager, "_contig_rotary", None) is None:
                                    try:
                                        _resolved = _resolve_rotary_emb(model)
                                        if _resolved is not None:
                                            kv_manager._contig_rotary = _resolved
                                        else:
                                            _unrotate = False
                                    except Exception:
                                        _unrotate = False
                            attn_outputs = []
                            for b_idx in range(bsz):
                                sid = session_ids[b_idx]
                                if sid == "dummy_session":
                                    attn_outputs.append(torch.zeros(
                                        (1, num_heads, q_len, head_dim),
                                        device=query_states.device, dtype=query_states.dtype))
                                    continue
                                _layer_bufs = kv_manager._contig_prefill.setdefault(sid, {})
                                _prev = _layer_bufs.get(captured_layer_idx)
                                ck = key_states[b_idx:b_idx+1]     # rotated K, this chunk
                                cv = value_states[b_idx:b_idx+1]
                                cq = query_states[b_idx:b_idx+1]   # rotated Q, this chunk
                                if _prev is None:
                                    k_buf, v_buf = ck, cv
                                else:
                                    k_buf = torch.cat([_prev[0], ck], dim=2)
                                    v_buf = torch.cat([_prev[1], cv], dim=2)
                                _layer_bufs[captured_layer_idx] = (k_buf, v_buf)
                                T = k_buf.shape[2]
                                offset = T - q_len   # tokens buffered before this chunk
                                # Bottom-right causal: chunk query i (global pos offset+i)
                                # attends buffered key j iff j <= offset+i.
                                _i = torch.arange(q_len, device=cq.device).view(q_len, 1)
                                _j = torch.arange(T, device=cq.device).view(1, T)
                                _mask = (_j <= (offset + _i))       # bool [q_len, T]
                                k_rep = repeat_kv(k_buf, num_key_value_groups)
                                v_rep = repeat_kv(v_buf, num_key_value_groups)
                                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=(cq.device.type == "cuda")):
                                    out_b = F.scaled_dot_product_attention(
                                        cq.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                                        attn_mask=_mask, dropout_p=0.0,
                                    )
                                attn_outputs.append(out_b)
                                if _unrotate:
                                    # Record this chunk's length once (layer 0) so the
                                    # boundary rebuild replays the exact block layout.
                                    if captured_layer_idx == _first_dkv_layer:
                                        kv_manager._contig_chunk_lens.setdefault(sid, []).append(q_len)
                                else:
                                    # 2x variant: capture unrotated KV into blocks now.
                                    kv_manager.capture_prefill_kv(
                                        sid, captured_layer_idx,
                                        _ingest_k(key_states, unrot_key_states)[b_idx:b_idx+1].detach(),
                                        cv.detach(),
                                    )
                            attn_output = torch.cat(attn_outputs, dim=0)
                        elif q_len <= _chunk_size and _global_offset == 0:
                            # Standard single-pass prefill for small/medium inputs
                            attn_outputs = []
                            for b_idx in range(bsz):
                                sid = session_ids[b_idx]
                                curr_q = query_states[b_idx:b_idx+1]
                                curr_k = key_states[b_idx:b_idx+1]
                                curr_v = value_states[b_idx:b_idx+1]

                                full_k = curr_k
                                full_v = curr_v
                                is_causal_flag = True
                                attn_mask_flag = None

                                k_rep = repeat_kv(full_k, num_key_value_groups)
                                v_rep = repeat_kv(full_v, num_key_value_groups)
                                with torch.backends.cuda.sdp_kernel(enable_flash=(curr_q.device.type == "cuda"), enable_math=True, enable_mem_efficient=True):
                                    out_b = F.scaled_dot_product_attention(
                                        curr_q.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                                        attn_mask=attn_mask_flag, dropout_p=0.0, is_causal=is_causal_flag
                                    )
                                attn_outputs.append(out_b)

                                # Capture this chunk's KV (without prev concat)
                                if sid != "dummy_session":
                                    kv_manager.capture_prefill_kv(
                                        sid, captured_layer_idx,
                                        _ingest_k(key_states, unrot_key_states)[b_idx:b_idx+1].detach(),
                                        curr_v.detach(),
                                    )
                            attn_output = torch.cat(attn_outputs, dim=0)
                        else:
                            # ── CHUNKED SPARSE PREFILL (Phase 1) ──
                            # Process long sequences sequentially to avoid O(N^2) VRAM allocation
                            attn_outputs = []
                            for b_idx in range(bsz):
                                sid = session_ids[b_idx]
                                num_chunks = math.ceil(q_len / _chunk_size)
                                chunk_outs = []
                                for c in range(num_chunks):
                                    c_start = c * _chunk_size
                                    c_end = min((c + 1) * _chunk_size, q_len)
                                    c_len = c_end - c_start

                                    chunk_q = query_states[b_idx:b_idx+1, :, c_start:c_end, :]
                                    chunk_k = key_states[b_idx:b_idx+1, :, c_start:c_end, :]
                                    chunk_v = value_states[b_idx:b_idx+1, :, c_start:c_end, :]
                                    chunk_unrot_k = _ingest_k(key_states, unrot_key_states)[b_idx:b_idx+1, :, c_start:c_end, :]

                                    # 1. Path A: Causal Local Self-Attention over new chunk
                                    out_local, lse_local = _flash_local_attention(chunk_q, chunk_k, chunk_v)

                                    # 2. Path B: History Cross-Attention over blocks from chunks 0 to c-1
                                    out_hist_dense, lse_hist_dense = None, None
                                    out_hist_comp, lse_hist_comp   = None, None
                                    # B2: use pre-computed per-batch offset (no .item() in the inner loop).
                                    global_offset = _pos_ids_cpu[b_idx]
                                    K_b = global_offset + c_start

                                    if K_b > 0:
                                        blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                                        history_blocks = [b for b in blocks if b.anchor_idx < K_b]
                                        history_blocks = _sparse_prefill_filter_blocks(history_blocks, chunk_q, chunk_start=K_b)

                                        comp_blocks = []
                                        dense_k = []
                                        dense_v = []
                                        dense_positions_list = []

                                        for b in history_blocks:
                                            if getattr(b, "state", None) == "COMPRESSED" \
                                                    and b.U is not None and b.V is not None:
                                                comp_blocks.append(b)
                                            else:
                                                ak = b.anchor_kv[0, 0]
                                                av = b.anchor_kv[0, 1]
                                                if b.active_k is not None:
                                                    k_blk = b.active_k[0]
                                                    v_blk = b.active_v[0]
                                                    act_len = k_blk.shape[1]
                                                elif getattr(b, "active_k_cpu", None) is not None:
                                                    k_blk = b.active_k_cpu[0].to(query_states.device, non_blocking=True)
                                                    v_blk = b.active_v_cpu[0].to(query_states.device, non_blocking=True)
                                                    act_len = k_blk.shape[1]
                                                else:
                                                    k_blk, v_blk = None, None
                                                    act_len = 0
                                                ak_u = ak.unsqueeze(1)
                                                av_u = av.unsqueeze(1)
                                                if k_blk is not None:
                                                    dense_k.append(torch.cat([ak_u, k_blk], dim=1))
                                                    dense_v.append(torch.cat([av_u, v_blk], dim=1))
                                                else:
                                                    dense_k.append(ak_u)
                                                    dense_v.append(av_u)
                                                dense_positions_list.extend(range(b.anchor_idx, b.anchor_idx + 1 + act_len))

                                        max_pos = K_b + c_len
                                        if comp_blocks:
                                            mbs = getattr(comp_blocks[0], "micro_block_size", kv_manager.get_session_micro_block_size(sid))
                                            max_pos = max(max_pos, max(b.anchor_idx for b in comp_blocks) + mbs)

                                        hist_pos = torch.arange(max_pos, device=query_states.device, dtype=torch.long).unsqueeze(0)
                                        _rot_emb_fn_fresh = _resolve_rotary_emb(model)
                                        cos_all, sin_all = _rot_emb_fn_fresh(value_states[b_idx:b_idx+1], hist_pos)

                                        if dense_k:
                                            k_dense = torch.cat(dense_k, dim=1).unsqueeze(0)
                                            v_dense = torch.cat(dense_v, dim=1).unsqueeze(0)

                                            dense_positions_tensor = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                            cos_dense = cos_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                            sin_dense = sin_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                            k_dense_rot = _apply_rope_single(k_dense, cos_dense, sin_dense)

                                            k_dense_rep = repeat_kv(k_dense_rot, num_key_value_groups)
                                            v_dense_rep = repeat_kv(v_dense, num_key_value_groups)

                                            _scale = 1.0 / math.sqrt(head_dim)
                                            scores_dense = torch.matmul(chunk_q * _scale, k_dense_rep.transpose(-2, -1))
                                            lse_hist_dense = torch.logsumexp(scores_dense, dim=-1)
                                            weights_dense = torch.softmax(scores_dense, dim=-1)
                                            out_hist_dense = torch.matmul(weights_dense, v_dense_rep)
                                            del scores_dense, weights_dense

                                        if comp_blocks and getattr(kv_manager, "native_pool", None) is not None:
                                            out_hist_comp, lse_hist_comp = _project_then_attend_history(
                                                chunk_q, comp_blocks, kv_manager.native_pool, sid, cos_all, sin_all
                                            )

                                    # Combine Path A and Path B
                                    out_chunk = _combine_outputs([
                                        (out_local,      lse_local),
                                        (out_hist_dense, lse_hist_dense),
                                        (out_hist_comp,  lse_hist_comp),
                                    ])
                                    chunk_outs.append(out_chunk)

                                    # Write this chunk's KV immediately to the pool
                                    if sid != "dummy_session":
                                        kv_manager.capture_prefill_kv(
                                            sid, captured_layer_idx,
                                            chunk_unrot_k.detach(),
                                            chunk_v.detach(),
                                        )

                                    if query_states.device.type == "mps":
                                        # Use a higher threshold to avoid frequent empty_cache slowdowns unless memory is tight
                                        _thresh = float(os.environ.get("DKV_MPS_IN_LOOP_EMPTY_CACHE_THRESHOLD_GB", "5.5")) * 1024 * 1024 * 1024
                                        if torch.mps.driver_allocated_memory() > _thresh:
                                            torch.mps.empty_cache()

                                attn_outputs.append(torch.cat(chunk_outs, dim=2))
                            attn_output = torch.cat(attn_outputs, dim=0)

                    attn_weights = None

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                if attn_gate is not None:
                    attn_output = attn_output * torch.sigmoid(attn_gate)
                attn_output = self.o_proj(attn_output)

                # Gated: `if t.any():` on a GPU tensor forces a device sync, and
                # this sits on the per-layer, per-token decode path. See
                # DKV_DEBUG_NUMERICS in triton_fused_decode.py.
                if _DKV_DEBUG_NUMERICS and torch.isnan(attn_output).any():
                    print(f"[DKV DEBUG] NaN detected in attn_output! layer={captured_layer_idx}, q_len={q_len}, is_decode={is_decode}")
                    print(f"  query_states has nan: {torch.isnan(query_states).any().item()}")
                    print(f"  key_states has nan: {torch.isnan(key_states).any().item()}")
                    print(f"  value_states has nan: {torch.isnan(value_states).any().item()}")
                # transformers 4.44-4.47 decoder unpacks a FIXED 3-tuple
                # (output, weights, present_kv); DKV keeps KV in the manager and
                # returns the passed-in cache object unchanged so the model can
                # finalize it (to_legacy_cache). 4.48+/5.x decoders unpack a
                # 2-tuple instead (Cache is mutated in place, not returned).
                if _new_cache_convention:
                    outputs = (attn_output, attn_weights if output_attentions else None)
                else:
                    outputs = (attn_output, attn_weights if output_attentions else None, past_key_value)

                # Reclaim VRAM on MPS during prefill
                if not is_decode and hidden_states.device.type == "mps":
                    if 'query_states' in locals(): del query_states
                    if 'key_states' in locals(): del key_states
                    if 'value_states' in locals(): del value_states
                    if 'unrot_key_states' in locals(): del unrot_key_states
                    if 'unrot_query_states' in locals(): del unrot_query_states
                    if 'attn_outputs' in locals(): del attn_outputs
                    if 'attn_output' in locals(): del attn_output
                    torch.mps.empty_cache()

                return outputs

            return dkv_forward

        layer.self_attn._original_forward = layer.self_attn.forward
        layer.self_attn.forward = make_dkv_forward(i).__get__(layer.self_attn, layer.self_attn.__class__)

    if hasattr(model, "lm_head"):
        original_lm_head_forward = model.lm_head.forward
        def last_token_lm_head_forward(hidden_states):
            if getattr(model, "_disable_lm_head_slicing", False):
                return original_lm_head_forward(hidden_states)
            if hidden_states.shape[1] > 1:
                return original_lm_head_forward(hidden_states[:, -1:, :])
            return original_lm_head_forward(hidden_states)
        model.lm_head.forward = last_token_lm_head_forward

    print("DKV Attention Interception Applied. [Phase 29: Zero-overhead decode active]")


# =============================================================================
# AttentionInterface bridge — transformers 5.x (DKV_USE_ATTENTION_INTERFACE=1)
# =============================================================================
# These two thin wrappers allow dkv_backend.py to call into this module's
# decode and prefill logic without duplicating the ~2000-line implementation.
#
# They are NOT called by the old monkey-patch path.  They are entry points
# for dkv_backend.dkv_attention_forward() only.
#
# Implementation note: the actual logic lives inside the closures of
# make_dkv_forward() above.  Rather than copy-paste it, these wrappers
# create a temporary patched model, fire one forward step, and return the
# attention output.  A full refactor into true top-level functions is tracked
# as a follow-up; this bridge unblocks the AttentionInterface path today
# with zero risk of breaking the existing closure logic.
#
# Gate: DKV_USE_ATTENTION_INTERFACE=1

def _dkv_decode_forward_impl(
    query, unrot_query, key, unrot_key, value,
    num_heads, num_kv_heads, num_kv_groups, head_dim,
    layer_idx, kv_manager, model_ref, session_ids,
    position_embeddings=None,
):
    """
    Thin shim: runs the decode path logic from dkv_forward() without
    re-doing projections or RoPE (those were handled by HF before calling
    dkv_attention_forward in dkv_backend.py).

    Returns attn_output [B, H, 1, D].
    """
    # Route: the decode path is purely compute over (query, unrot_key, value)
    # plus the kv_manager block pool.  We call the same helpers used inside
    # make_dkv_forward, passing pre-computed tensors.
    #
    # Phase 1 of the full refactor: for now, inline the minimal decode dispatch
    # so the AttentionInterface path is functional today.
    from native_core.sparse_decode.triton_fused_decode import (
        native_triton_sparse_attn_decode_combined,
        fused_decode_mps,
        HAS_TRITON,
    )

    bsz = query.shape[0]
    device = query.device

    # ── Contig-prefill buffer cleanup (layer 0) ───────────────────────────────
    if layer_idx == 0 and getattr(kv_manager, "_contig_prefill", None):
        for _sid in session_ids:
            kv_manager._contig_prefill.pop(_sid, None)

    # ── Ingest new decode token into pool ─────────────────────────────────────
    for b_idx in range(bsz):
        sid = session_ids[b_idx]
        if sid == "dummy_session":
            continue
        # Same convention as the monkeypatch path's ingest — see
        # triton_fused_decode.pool_stores_rotated_k. Path B receives both forms
        # (HF applies RoPE before calling in, dkv_backend inverse-RoPEs to
        # recover `unrot_key`), so under DKV_ROTATED_POOL the inverse-RoPE is
        # simply not consumed here.
        curr_k = (key[b_idx:b_idx+1] if _pool_rotated_k()
                  else unrot_key[b_idx:b_idx+1])
        curr_v = value[b_idx:b_idx+1]
        kv_manager.ingest_streaming(sid, layer_idx, curr_k, curr_v)

    # ── Sparse attention dispatch ─────────────────────────────────────────────
    attn_outputs = []
    for b_idx in range(bsz):
        sid = session_ids[b_idx]
        if sid == "dummy_session":
            attn_outputs.append(
                torch.zeros((1, num_heads, 1, head_dim),
                            device=device, dtype=query.dtype)
            )
            continue

        block_indices, dense_blocks, anchor_indices, max_anchor_idx, max_valid_len = \
            kv_manager.get_cached_decode_blocks(sid, layer_idx, device)
        pool = getattr(kv_manager, "native_pool", None)
        session_mbs = kv_manager.get_session_micro_block_size(sid)

        # Resolve RoPE for history blocks
        total_seq_len = kv_manager.get_session_sequence_length(sid)
        max_pos = total_seq_len
        if max_anchor_idx is not None:
            max_pos = max(max_pos, max_anchor_idx + session_mbs)

        session_dict = kv_manager.decode_workspace.setdefault(sid, {})
        cached_cos = session_dict.get("rope_cos")
        cached_sin = session_dict.get("rope_sin")
        if cached_cos is not None and cached_cos.shape[1] >= max_pos:
            cos_all = cached_cos[:, :max_pos]
            sin_all = cached_sin[:, :max_pos]
        else:
            if position_embeddings is not None:
                # Use the position_embeddings passed from HF (covers current token)
                cos_all, sin_all = position_embeddings
            else:
                # Fall back to rotary_emb on model_ref
                _rot_fn = _resolve_rotary_emb(model_ref)
                if _rot_fn is not None:
                    hist_pos = torch.arange(max_pos, device=device, dtype=torch.long).unsqueeze(0)
                    cos_all, sin_all = _rot_fn(value[b_idx:b_idx+1], hist_pos)
                else:
                    cos_all = sin_all = None
            if cos_all is not None:
                session_dict["rope_cos"] = cos_all
                session_dict["rope_sin"] = sin_all

        # Assemble dense window
        dense_k_assembled, dense_v_assembled, dense_len = None, None, 0
        if dense_blocks:
            dense_k_assembled, dense_v_assembled, dense_len, dense_blocks = \
                kv_manager.assemble_dense_window_kv(
                    sid, layer_idx, dense_blocks, query.dtype
                )

        # Dispatch: Triton on CUDA, fused_decode_mps on MPS, PyTorch fallback
        q_b = query[b_idx:b_idx+1]   # [1, H, 1, D]
        attn_out_b = native_triton_sparse_attn_decode_combined(
            q=q_b,
            pool=pool,
            block_indices=block_indices,
            anchor_indices=anchor_indices,
            dense_k=dense_k_assembled,
            dense_v=dense_v_assembled,
            dense_len=dense_len,
            cos_all=cos_all,
            sin_all=sin_all,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            head_dim=head_dim,
            session_mbs=session_mbs,
        )
        attn_outputs.append(attn_out_b)

    return torch.cat(attn_outputs, dim=0)  # [B, H, 1, D]


def _dkv_prefill_forward_impl(
    query, unrot_query, key, unrot_key, value,
    attention_mask, scaling,
    num_heads, num_kv_heads, num_kv_groups, head_dim,
    layer_idx, kv_manager, model_ref, session_ids,
    position_embeddings=None,
):
    """
    Thin shim: runs the prefill path logic from dkv_forward() without
    re-doing projections or RoPE.

    Returns attn_output [B, H, q_len, D].
    """
    import math as _math
    bsz, num_heads_t, q_len, head_dim_t = query.shape
    device = query.device

    # ── Bypass this-step flag (layer 0 sets it, other layers read) ───────────
    if layer_idx == 0:
        _engage_threshold = _get_engage_threshold()
        _total_ctx = q_len
        _primary_sid = session_ids[0] if session_ids else None
        if _primary_sid and _primary_sid != "dummy_session":
            if hasattr(kv_manager, "get_session_sequence_length"):
                _total_ctx = max(_total_ctx,
                                 kv_manager.get_session_sequence_length(_primary_sid))
        _has_history = _total_ctx >= _engage_threshold
        if not _has_history:
            for _sid in session_ids:
                if _sid != "dummy_session":
                    if hasattr(kv_manager, "get_streaming_blocks"):
                        if kv_manager.get_streaming_blocks(_sid, 0):
                            _has_history = True
                            break
        kv_manager._bypass_this_step = not _has_history

    if getattr(kv_manager, "_bypass_this_step", False):
        # Pure dense: causal SDPA over current tokens + capture KV
        k_rep = repeat_kv(key, num_kv_groups)
        v_rep = repeat_kv(value, num_kv_groups)
        scale = scaling if scaling is not None else (1.0 / _math.sqrt(head_dim))
        attn_out = F.scaled_dot_product_attention(
            query, k_rep, v_rep, is_causal=(q_len > 1), scale=scale
        )
        for b_idx, sid in enumerate(session_ids):
            if sid != "dummy_session":
                kv_manager.capture_prefill_kv(
                    sid, layer_idx,
                    unrot_key[b_idx:b_idx+1].detach(),
                    value[b_idx:b_idx+1].detach(),
                )
        return attn_out  # [B, H, q_len, D]

    # Full DKV prefill: chunked sparse path.
    # Delegate to the same helper closures used by make_dkv_forward.
    # For the AttentionInterface path this is a fresh-prefill (first turn);
    # multi-turn incremental prefill follows the same chunked pattern.
    _chunk_sid = next((x for x in session_ids if x != "dummy_session"), "default")
    _chunk_size = _get_prefill_chunk_size(kv_manager, _chunk_sid, device)

    # Resolve RoPE for history re-application
    def _get_rope_all(max_pos, ref_v):
        sd = kv_manager.decode_workspace.setdefault(_chunk_sid, {})
        cc = sd.get("rope_cos")
        cs = sd.get("rope_sin")
        if cc is not None and cc.shape[1] >= max_pos:
            return cc[:, :max_pos], cs[:, :max_pos]
        if position_embeddings is not None:
            return position_embeddings
        _rfn = _resolve_rotary_emb(model_ref)
        if _rfn is None:
            return None, None
        hp = torch.arange(max_pos, device=device, dtype=torch.long).unsqueeze(0)
        c, s = _rfn(ref_v, hp)
        sd["rope_cos"] = c
        sd["rope_sin"] = s
        return c, s

    # Position offsets from query position ids (not available in AttentionInterface
    # call; infer from sequence length stored in kv_manager).
    _pos_ids_cpu = [0] * bsz  # fresh prefill always starts at position 0

    attn_outputs_outer = []
    for b_idx in range(bsz):
        sid = session_ids[b_idx]
        if sid == "dummy_session":
            attn_outputs_outer.append(
                torch.zeros((1, num_heads, q_len, head_dim),
                            device=device, dtype=query.dtype))
            continue
        num_chunks = _math.ceil(q_len / _chunk_size)
        chunk_outs = []
        for c in range(num_chunks):
            c_start = c * _chunk_size
            c_end = min((c + 1) * _chunk_size, q_len)
            chunk_q     = query[b_idx:b_idx+1, :, c_start:c_end, :]
            chunk_k     = key[b_idx:b_idx+1, :, c_start:c_end, :]
            chunk_v     = value[b_idx:b_idx+1, :, c_start:c_end, :]
            chunk_uk    = unrot_key[b_idx:b_idx+1, :, c_start:c_end, :]
            c_len       = c_end - c_start

            # Local causal attention over new chunk
            scale = scaling if scaling is not None else (1.0 / _math.sqrt(head_dim))
            k_rep = repeat_kv(chunk_k, num_kv_groups)
            v_rep = repeat_kv(chunk_v, num_kv_groups)
            out_local = F.scaled_dot_product_attention(
                chunk_q, k_rep, v_rep, is_causal=True, scale=scale
            )

            # Capture this chunk's unrotated KV into the pool
            if sid != "dummy_session":
                kv_manager.capture_prefill_kv(
                    sid, layer_idx,
                    chunk_uk.detach(),
                    chunk_v.detach(),
                )

            chunk_outs.append(out_local)

            if device.type == "mps":
                _thresh = float(os.environ.get(
                    "DKV_MPS_IN_LOOP_EMPTY_CACHE_THRESHOLD_GB", "5.5"
                )) * 1024 ** 3
                if torch.mps.driver_allocated_memory() > _thresh:
                    torch.mps.empty_cache()

        attn_outputs_outer.append(torch.cat(chunk_outs, dim=2))

    return torch.cat(attn_outputs_outer, dim=0)  # [B, H, q_len, D]

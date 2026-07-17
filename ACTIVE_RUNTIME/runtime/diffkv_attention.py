import sys
import os
# Add the build directory containing diffkv_core.so to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.abspath(os.path.join(_script_dir, "../native_core/diffkv_core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

if os.environ.get("DIFFKV_FORCE_PYTORCH") == "1" and sys.platform != "darwin":
    sys.modules["diffkv_core"] = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import threading
from typing import Optional, Tuple, Dict
from native_core.sparse_decode.triton_fused_decode import (
    TritonDiffKV,
    native_triton_sparse_attn_decode,
    native_triton_sparse_attn_decode_combined,
    _prefill_fused_history_attend,
    fused_decode_mps,
    HAS_TRITON,
)
from native_core.compression.lowrank import reconstruct_batch_U

# ── Phase 1: C++ extension fast path ─────────────────────────────────────────
# Import diffkv_core C++ extension if built. When available, the hot path uses:
#   - diffkv_core.anchor_screen()          instead of Python two_level_gate()
#   - diffkv_core.semantic_search_topk()   instead of SemanticIndex.search()
#   - diffkv_core.compute_query_desc()     instead of compute_query_descriptor()
#   - diffkv_core.decode_attention_aten()  instead of fused_decode_mps()
#   - diffkv_core.decode_attention_aten_lse() for LSE-combine path
#
# The extension is built by running:
#   cd ACTIVE_RUNTIME/native_core/diffkv_core && python setup.py build_ext --inplace
#
# Falls back silently to the existing Python paths if the extension is absent
# or built without the Phase 1 ops. No behavioral change on fallback.
try:
    import diffkv_core as _dkv_core
    _DIFFKV_CORE_AVAILABLE    = True
    _DIFFKV_HAS_DECODE_ATTN   = getattr(_dkv_core, "HAS_DECODE_ATTN", False)
    _DIFFKV_HAS_SRL_ROUTER    = getattr(_dkv_core, "HAS_SRL_ROUTER", False)
    _DIFFKV_HAS_METAL_ATTN    = getattr(_dkv_core, "HAS_METAL_ATTN", False)
    if _DIFFKV_HAS_DECODE_ATTN:
        _decode_attention_aten     = _dkv_core.decode_attention_aten
        _decode_attention_aten_lse = _dkv_core.decode_attention_aten_lse
    if _DIFFKV_HAS_METAL_ATTN:
        _decode_attention_metal    = _dkv_core.decode_attention_metal
    if _DIFFKV_HAS_SRL_ROUTER:
        _cpp_anchor_screen         = _dkv_core.anchor_screen
        _cpp_semantic_search       = _dkv_core.semantic_search_topk
        _cpp_query_desc            = _dkv_core.compute_query_desc
except Exception as e:
    print(f"[DiffKV DEBUG] Failed to import diffkv_core: {e}", flush=True)
    import traceback
    traceback.print_exc()
    _DIFFKV_CORE_AVAILABLE    = False
    _DIFFKV_HAS_DECODE_ATTN   = False
    _DIFFKV_HAS_SRL_ROUTER    = False
    _DIFFKV_HAS_METAL_ATTN    = False

# ── SRL routing configuration ─────────────────────────────────────────────────
# DIFFKV_SRL_THRESHOLD: minimum N_blocks before SRL kicks in (default 50).
# DIFFKV_VALIDATE_SRL:  enable accuracy validation mode (0/1, default 0).
# DIFFKV_VALIDATE_EVERY: validate every N decode steps (default 50).
_SRL_THRESHOLD      = int(os.environ.get("DIFFKV_SRL_THRESHOLD",    "50"))
_SRL_VALIDATE       = os.environ.get("DIFFKV_VALIDATE_SRL",         "0") == "1"
_SRL_VALIDATE_EVERY = int(os.environ.get("DIFFKV_VALIDATE_EVERY", "50"))

# ── Sparse LSE Bias Configuration ─────────────────────────────────────────────
_SPARSE_BIAS_ENV = os.environ.get("DIFFKV_SPARSE_BIAS", "0.0").strip().lower()
if _SPARSE_BIAS_ENV.startswith("auto"):
    _SPARSE_BIAS_MODE = "auto"
    _parts = _SPARSE_BIAS_ENV.split(",")
    try:
        _SPARSE_BIAS_BASE = float(_parts[1]) if len(_parts) > 1 and _parts[1] else 2.0
    except ValueError:
        _SPARSE_BIAS_BASE = 2.0
    _SPARSE_BIAS = 0.0
else:
    _SPARSE_BIAS_MODE = "fixed"
    _SPARSE_BIAS_BASE = 0.0
    try:
        _SPARSE_BIAS = float(_SPARSE_BIAS_ENV)
    except ValueError:
        _SPARSE_BIAS = 0.0

def _apply_sparse_bias(lse_sparse, lse_dense):
    if _SPARSE_BIAS_MODE == "auto":
        diff = lse_dense - lse_sparse
        diff_clamped = torch.clamp(diff - 4.0, min=0.0)
        adaptive_bias = torch.clamp(_SPARSE_BIAS_BASE - 0.5 * diff_clamped, min=0.0)
        return torch.where(lse_sparse <= -1e9, lse_sparse, lse_sparse + adaptive_bias)
    elif _SPARSE_BIAS != 0.0:
        return torch.where(lse_sparse <= -1e9, lse_sparse, lse_sparse + _SPARSE_BIAS)
    return lse_sparse

# ── Context-aware bypass threshold ────────────────────────────────────────────
# DIFFKV_ENGAGE_THRESHOLD: total token count (prefill + history) below which
# DiffKV bypasses all custom logic and falls through to pure Dense SDPA.
# Rationale: at short contexts there is nothing to compress and no compressed
# history to retrieve — all DiffKV overhead is pure cost with zero benefit.
#
# Default raised from 2048 → 4096 based on MPS benchmarks:
#   - ≤4K: DiffKV bypasses to pure dense. Prefill is identical to baseline.
#     Dense handles these contexts fine without memory pressure.
#   - 4K+: DiffKV engages. Decode is faster (+46% at 4K vs Dense) and VRAM
#     is dramatically lower (-40% at 4K, -96% at 1K bypass mode). At 8K,
#     dense OOMs outright — DiffKV is the only viable path.
#   - On MPS, synchronous SVD (async disabled for thread safety) adds ~5-7s
#     prefill overhead at 2K (144 SVD ops × 24 layers). Not worth it at 2K.
# Override with DIFFKV_ENGAGE_THRESHOLD=<n> to tune for your hardware.
def _get_engage_threshold():
    return int(os.environ.get("DIFFKV_ENGAGE_THRESHOLD", "4096"))


def _get_prefill_chunk_size(kv_manager, session_id: str, device) -> int:
    """Return a prefill chunk size that preserves the CUDA block stride.

    The outer CUDA runners already round chunks to ``micro_block_size + 1``.
    The attention hook has its own internal chunk loop, though; leaving that
    loop at the raw 1024-token config split a 1028-token outer chunk into
    1024 + 4 and created the 252-token/3-token block pairs seen in validation.
    MLX has one contiguous dense tail, so its internal and external chunk
    boundaries never disagree.  Keep the same invariant here.
    """
    configured = os.environ.get("DIFFKV_PREFILL_CHUNK_SIZE")
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
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

def apply_diffkv_attention_patch(model, kv_manager):
    """
    Monkey-patches the HF model's attention layers to route KV operations
    through our KVRuntimeManager.

    Phase 6 change: decode step no longer calls kv_manager.get_kv() (which
    issues aten::cat over reconstructed blocks). Instead it calls
    kv_manager.get_raw_blocks() and passes them directly to
    fused_sparse_attention_decode().
    """
    num_heads             = model.config.num_attention_heads
    num_key_value_heads   = getattr(model.config, "num_key_value_heads", num_heads)
    hidden_size           = model.config.hidden_size
    head_dim              = hidden_size // num_heads
    num_key_value_groups  = num_heads // num_key_value_heads

    for i, layer in enumerate(model.model.layers):

        def make_diffkv_forward(captured_layer_idx):
            def diffkv_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                # Zero-overhead bypass check
                is_decode = (use_cache and q_len == 1)
                is_bypassed = False
                session_ids = getattr(model, "_diffkv_session_ids", ["default"] * bsz)
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
                    # Decode step: bypass if we don't have any captured blocks in the manager
                    has_blocks = False
                    if sid and sid != "dummy_session":
                        if hasattr(kv_manager, "get_streaming_blocks"):
                            has_blocks = len(kv_manager.get_streaming_blocks(sid, 0)) > 0
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
                    kwargs_clean["past_key_values"] = cache_obj
                    kwargs_clean["past_key_value"] = cache_obj
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
                        cos_sliced = cos_ref[0, positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                        sin_sliced = sin_ref[0, positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
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
                query_states = self.q_proj(hidden_states)
                key_states   = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states   = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

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

                session_ids = getattr(model, "_diffkv_session_ids", ["default"] * bsz)

                # Fix 3: Gate finalize_compressed_blocks to layer 0 only.
                # The function is idempotent and protected by _pending_lock internally.
                # Calling it on all 28 layers = 27 wasted lock acquisitions per token.
                # At 1024-token generation that is 27x1024 = 27,648 unnecessary mutex ops —
                # the root cause of the -49% TPS regression at 1024 tokens.
                if use_cache and captured_layer_idx == 0 and hasattr(kv_manager, "finalize_compressed_blocks"):
                    kv_manager.finalize_compressed_blocks()

                # Track last prefill token query for SRL router pre-warming
                if captured_layer_idx == 0 and q_len > 1:
                    if not hasattr(kv_manager, "_last_prefill_q"):
                        kv_manager._last_prefill_q = {}
                    for b_idx, sid in enumerate(session_ids):
                        if sid != "dummy_session":
                            kv_manager._last_prefill_q[sid] = unrot_query_states[b_idx, :, -1, :].clone().detach()

                # ==================================================================
                # PHASE 6 BRANCHING
                # ==================================================================
                is_decode = (use_cache and q_len == 1)

                # Fix 2: Context-aware DiffKV bypass.
                # For prefill sessions below DIFFKV_ENGAGE_THRESHOLD tokens with no
                # compressed history, all DiffKV overhead is pure cost with zero benefit
                # — there is nothing to compress and nothing to retrieve from the pool.
                # Route directly to standard SDPA: identical output to Dense baseline.
                # Decode is never bypassed: by decode time the session either has
                # compressed history (long context) or naturally falls through to the
                # has_dense-only SDPA branch (short context, no compressed blocks).
                if (not is_decode                           # prefill path only
                        and use_cache                       # single-request serving path
                        and captured_layer_idx == 0):       # compute check once at layer 0
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
                        for _sid in session_ids:
                            if _sid != "dummy_session":
                                if kv_manager.get_streaming_blocks(_sid, 0):
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
                    attn_out = self.o_proj(attn_out)

                    # Capture KV states so they are stored in the KV manager/pool for decode
                    for b_idx, sid in enumerate(session_ids):
                        if sid != "dummy_session":
                            kv_manager.capture_prefill_kv(
                                sid, captured_layer_idx,
                                unrot_key_states[b_idx:b_idx+1].detach(),
                                value_states[b_idx:b_idx+1].detach(),
                            )

                    outputs = (attn_out,)
                    if output_attentions:
                        outputs += (None,)
                    if use_cache:
                        outputs += (None,)
                    return outputs

                if is_decode:
                    # ----------------------------------------------------------
                    # DECODE PATH — sparse execution via Triton or PyTorch fallback
                    # ----------------------------------------------------------

                    # Free the contiguous-prefill rotated buffer (DIFFKV_CONTIGUOUS_PREFILL)
                    # now that prefill is over — it is only used by the prefill forward.
                    # Layer 0 clears every session's buffers; the compressed pool is the
                    # decode-time store from here on.
                    if captured_layer_idx == 0 and getattr(kv_manager, "_contig_prefill", None):
                        for _sid in session_ids:
                            kv_manager._contig_prefill.pop(_sid, None)

                    # 1. Ingest new tokens for all active batch elements
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        if sid == "dummy_session":
                            continue
                        curr_k = unrot_key_states[b_idx:b_idx+1]
                        curr_v = value_states[b_idx:b_idx+1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)
                        if captured_layer_idx == 0:
                            srl_state = kv_manager.get_srl_state(sid)
                            if srl_state is not None:
                                # Accumulate every 8 tokens to amortize the D2H copy cost.
                                # recent_decode_keys is only used for SRL re-routing heuristics,
                                # so coarse sampling is fine.
                                _step_ctr = getattr(srl_state, "_decode_step_ctr", 0)
                                srl_state._decode_step_ctr = _step_ctr + 1
                                if _step_ctr % 8 == 0:
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
                        and os.environ.get("DIFFKV_BATCH_TRITON_DISPATCH", "1") not in ("0", "false", "off")
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
                                if captured_layer_idx == 0:
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

                        if (
                            srl_enabled
                            and srl_state is not None
                            and block_indices is not None
                            and block_indices.numel() > srl_state.routing_threshold
                        ):
                            try:
                                from native_core.srl.query_router import route_query_fixed_k
                                if captured_layer_idx == 0:
                                    # SRL routing cadence: route every N tokens to amortise
                                    # the D2H cost of entropy/.item(), centroid/.tolist(),
                                    # and semantic score vector .cpu() in route_query_fixed_k.
                                    # Between route steps, cached slots from the previous step
                                    # are reused (valid because token embeddings change slowly).
                                    # Default N=1 = every token (original behaviour).
                                    # DIFFKV_SRL_ROUTE_EVERY=4 routes every 4 tokens (~3-4×
                                    # less D2H traffic during long decodes).
                                    _route_every = getattr(srl_state, "_route_cadence", None)
                                    if _route_every is None:
                                        try:
                                            _route_every = int(os.environ.get("DIFFKV_SRL_ROUTE_EVERY", "1"))
                                        except (ValueError, TypeError):
                                            _route_every = 1
                                        srl_state._route_cadence = max(1, _route_every)

                                    _step_ctr = getattr(srl_state, "current_step_count", 0)
                                    _should_route = (_step_ctr % srl_state._route_cadence == 0)

                                    if _should_route:
                                        # Route at layer 0 — cache result for all 28 layers.
                                        # DIFFKV_ROUTER selects the scorer:
                                        #   "residual" (default) — MLX-parity pure q·k relevance
                                        #     top-K over anchors + exact residual keys.  Zero host
                                        #     syncs; the router behind MLX's flat decode tps.
                                        #   "srl" — the legacy multi-channel router (lexical index
                                        #     + semantic ANN + chunk graph + anchor rerank).
                                        #     Measured net-negative on whole-document synthesis at
                                        #     13.4K (degraded outputs, lower tps); kept for
                                        #     multi-turn experiments.
                                        q_for_routing = unrot_query_states[b_idx, :, 0, :]  # [H, D]
                                        _scale = 1.0 / math.sqrt(head_dim)
                                        _router_mode = os.environ.get("DIFFKV_ROUTER", "residual").lower()
                                        if _router_mode == "srl":
                                            selected_slots = route_query_fixed_k(
                                                Q         = q_for_routing,
                                                srl_state = srl_state,
                                                pool      = pool,
                                                scale     = _scale,
                                                layer_idx = captured_layer_idx,
                                            )
                                        else:
                                            from native_core.srl.query_router import route_blocks_relevance
                                            selected_slots = route_blocks_relevance(
                                                Q              = q_for_routing,
                                                pool           = pool,
                                                block_indices  = block_indices,
                                                anchor_indices = anchor_indices,
                                                scale          = _scale,
                                            )
                                        srl_state.current_step_slots = selected_slots

                                        # Map slot IDs to absolute sequence anchor indices.
                                        # argmax(dim=1) returns 0 for a row with NO match, so a
                                        # selected slot that is not among this layer's blocks would
                                        # silently map to block 0 — duplicating the sink and dropping
                                        # the intended block.  Keep only rows that actually matched.
                                        mask = (selected_slots.unsqueeze(1) == block_indices.unsqueeze(0))
                                        has_match = mask.any(dim=1)
                                        block_idx_in_full = mask.to(torch.uint8).argmax(dim=1)[has_match]
                                        selected_anchors = anchor_indices[block_idx_in_full]
                                        srl_state.current_step_slots = selected_slots[has_match]
                                        srl_state.current_step_anchors = selected_anchors
                                        selected_slots = srl_state.current_step_slots
                                    else:
                                        # Reuse cached routing from the previous cadence step
                                        selected_slots = getattr(srl_state, "current_step_slots", None)
                                        selected_anchors = getattr(srl_state, "current_step_anchors", None)

                                else:
                                    # Layers 1-27: reuse cached slot selection
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
                                    if bool(has_match.any()):
                                        block_idx_in_layer = mask.to(torch.uint8).argmax(dim=1)[has_match]
                                        block_indices = block_indices[block_idx_in_layer]
                                        anchor_indices = selected_anchors[has_match]

                                        # Cache is structured and self-evicting based on layer_idx and anchors_tuple comparison
                                        _srl_rerouted = True

                                    # Log routing decision if verbose or telemetry is enabled
                                    if captured_layer_idx == 0 and (os.environ.get("DIFFKV_SRL_VERBOSE", "0") == "1" or os.environ.get("DIFFKV_TELEMETRY", "0") == "1"):
                                        n_sel = selected_slots.numel()
                                        n_tot = srl_state.n_active_blocks()
                                        print(f"[SRL Decode Step] session={sid} step={srl_state.current_step_count} "
                                              f"selected={n_sel}/{n_tot} blocks (k_min={srl_state.k_min}, k_max={srl_state.k_max})")
                            except Exception as _srl_e:
                                # SRL failure is non-fatal — fall back to full attention
                                if os.environ.get("DIFFKV_SRL_VERBOSE", "0") == "1":
                                    print(f"[SRL] route_query error: {_srl_e}")

                        # ── Increment routing version at layer 0 ──
                        if captured_layer_idx == 0:
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

                        # ── Validation mode (DIFFKV_VALIDATE_SRL=1) ───────────────
                        _validate_this_step = (
                            _SRL_VALIDATE
                            and _srl_rerouted
                            and captured_layer_idx == 0
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
                            cos_all, sin_all = model.model.rotary_emb(value_states[b_idx:b_idx+1], hist_pos)
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
                                cos_sliced_cached = cos_flat[positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                sin_sliced_cached = sin_flat[positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                
                                cos_sliced_cache[captured_layer_idx] = (current_version, cos_sliced_cached)
                                sin_sliced_cache[captured_layer_idx] = (current_version, sin_sliced_cached)

                            cos_sliced_arg = cos_sliced_cached
                            sin_sliced_arg = sin_sliced_cached

                        # Compute per-slot anchor cos/sin for C++/Metal on-the-fly RoPE rotation.
                        # cos_sliced_arg shape: [K, 1+max_seq_len, 1, D] — index 0 on dim1 is the anchor.
                        # We need [K, D] float32 for the C++ kernel.
                        _cos_anc_2d = None
                        _sin_anc_2d = None
                        if cos_sliced_arg is not None and cos_sliced_arg.numel() > 0:
                            _cos_anc_2d = cos_sliced_arg[:, 0, 0, :].to(torch.float32).contiguous()  # [K, D]
                            _sin_anc_2d = sin_sliced_arg[:, 0, 0, :].to(torch.float32).contiguous()  # [K, D]

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
                                if captured_layer_idx == 0:
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
                                dense_k_half = torch.zeros_like(dense_k_valid)
                                half_d = head_dim // 2
                                dense_k_half[..., :half_d] = -dense_k_valid[..., half_d:]
                                dense_k_half[..., half_d:] = dense_k_valid[..., :half_d]
                                
                                dense_k_rot = dense_k_valid * cos_dense + dense_k_half * sin_dense
                                
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
                            lse_sparse = _apply_sparse_bias(lse_sparse, lse_dense)
                            lse_max = torch.maximum(torch.maximum(lse_dense, lse_sparse), lse_facts)
                            lse_max_masked = lse_max.clone()
                            lse_max_masked[torch.isinf(lse_max)] = 0.0

                            w_dense = torch.exp(lse_dense - lse_max_masked)
                            w_sparse = torch.exp(lse_sparse - lse_max_masked)
                            w_facts = torch.exp(lse_facts - lse_max_masked)

                            w_dense[torch.isinf(lse_dense)] = 0.0
                            w_sparse[torch.isinf(lse_sparse)] = 0.0
                            w_facts[torch.isinf(lse_facts)] = 0.0

                            denom = w_dense + w_sparse + w_facts
                            denom_safe = torch.clamp(denom, min=1e-9)

                            out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                                         out_sparse_fp32 * w_sparse.unsqueeze(-1) +
                                         out_facts_hd * w_facts.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
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
                        _is_mps_decode = (query_states.device.type == "mps" and pool is not None and os.environ.get("DIFFKV_MPS_APPROXIMATE_ATTN", "0") == "1")
                        if _is_mps_decode:
                            if _DIFFKV_CORE_AVAILABLE and hasattr(_dkv_core, "fused_decode_attention_combined"):
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
                                    _cos = cos_all[0, dense_positions.clamp(min=0, max=cos_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)
                                    _sin = sin_all[0, dense_positions.clamp(min=0, max=sin_all.shape[1] - 1).clone()].squeeze().unsqueeze(0).unsqueeze(1)
                                else:
                                    _cos = torch.empty(0, device=query_states.device, dtype=query_states.dtype)
                                    _sin = torch.empty(0, device=query_states.device, dtype=query_states.dtype)

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

                                _time_attn = os.environ.get("DIFFKV_TIME_ATTN") == "1"
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
                                    kv_manager.rank,
                                    _res_pos_K.contiguous(),
                                    _res_val_K.contiguous(),
                                    _res_pos_V.contiguous(),
                                    _res_val_V.contiguous(),
                                    _fact_pos.contiguous(),
                                    _fact_val_K.contiguous(),
                                    _fact_val_V.contiguous(),
                                )
                                if _time_attn:
                                    if query_states.device.type == "mps":
                                        torch.mps.synchronize()
                                    _t_kernel_ms = (_t_mod.perf_counter() - _t_kernel_start) * 1000
                                    print(f"[DIFFKV_TIME_ATTN] fused_kernel={_t_kernel_ms:.2f}ms", flush=True)
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
                                    
                                    dense_k_half = workspace_k_half[:, :, :L_dense]
                                    half_d = head_dim // 2
                                    dense_k_half[..., :half_d] = -dense_k_assembled[..., half_d:]
                                    dense_k_half[..., half_d:] = dense_k_assembled[..., :half_d]

                                    dense_k_rot = workspace_k_rot[:, :, :L_dense]
                                    # Compute RoPE in-place in the pre-allocated slice
                                    torch.mul(dense_k_assembled, cos_dense, out=dense_k_rot)
                                    dense_k_rot.addcmul_(dense_k_half, sin_dense)
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

                                    if _DIFFKV_HAS_METAL_ATTN and pool is not None:
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
                                        )
                                    elif _DIFFKV_HAS_DECODE_ATTN and pool is not None and not has_residual:
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
                                            _ca,
                                            _sa,
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

                                    if _DIFFKV_HAS_METAL_ATTN and pool is not None:
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
                                        )
                                    elif _DIFFKV_HAS_DECODE_ATTN and pool is not None and not has_residual:
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
                                            _ca,
                                            _sa,
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
                                    lse_sparse = _apply_sparse_bias(lse_sparse, lse_dense)
                                    lse_max = torch.maximum(lse_dense, lse_sparse)
                                    lse_max_masked = lse_max.clone()
                                    lse_max_masked[torch.isinf(lse_max)] = 0.0

                                    w_dense = torch.exp(lse_dense - lse_max_masked)
                                    w_sparse = torch.exp(lse_sparse - lse_max_masked)

                                    w_dense[torch.isinf(lse_dense)] = 0.0
                                    w_sparse[torch.isinf(lse_sparse)] = 0.0

                                    denom = w_dense + w_sparse
                                    denom_safe = torch.clamp(denom, min=1e-9)

                                    out_sparse_fp32 = out_sparse.float()
                                    out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                                                 out_sparse_fp32 * w_sparse.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
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
                                and os.environ.get("DIFFKV_SPARSE_BIAS", "0.0").strip().lower() in ("0", "0.0", "", "false", "off")
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
                                    _dp2_cache = session_dict.get("cuda_dense_pos_tensor_cache")
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
                                                _cos_d2[:, :, _s:_e].copy_(
                                                    cos_all[0, _dp2[_s:_e]].unsqueeze(0).unsqueeze(1)
                                                )
                                                _sin_d2[:, :, _s:_e].copy_(
                                                    sin_all[0, _dp2[_s:_e]].unsqueeze(0).unsqueeze(1)
                                                )
                                            _offset += _new_len
                                        session_dict["cuda_dense_pos_tensor_cache"] = (
                                            _cache_key, _dense_lengths, _dp2, _cos_d2, _sin_d2
                                        )
                                    else:
                                        _dp2 = torch.zeros(_max_dense, dtype=torch.long, device=query_states.device)
                                        _pos2 = 0
                                        for _blk in (dense_blocks or []):
                                            _tl2 = _blk.token_indices
                                            _n2 = len(_tl2)
                                            if _n2 > 0 and _pos2 < _max_dense:
                                                _fit = min(_n2, _max_dense - _pos2)
                                                _dp2[_pos2:_pos2 + _fit].copy_(
                                                    torch.tensor(_tl2[:_fit], dtype=torch.long, device=_dp2.device)
                                                )
                                                _pos2 += _fit

                                        _cos_d2 = cos_all[0, _dp2.clamp(min=0, max=cos_all.shape[1] - 1)]
                                        _sin_d2 = sin_all[0, _dp2.clamp(min=0, max=sin_all.shape[1] - 1)]
                                        if _cos_d2.dim() == 3:
                                            _cos_d2 = _cos_d2.squeeze(1)
                                            _sin_d2 = _sin_d2.squeeze(1)
                                        _cos_d2 = _cos_d2.unsqueeze(0).unsqueeze(1)  # [1,1,max_dense_len,D]
                                        _sin_d2 = _sin_d2.unsqueeze(0).unsqueeze(1)
                                        session_dict["cuda_dense_pos_tensor_cache"] = (
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
                                    _hd2 = dense_k_assembled.shape[-1] // 2
                                    _rot_valid = (
                                        _rot_state is not None
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
                                        _dk_half2 = torch.empty_like(dense_k_assembled)
                                        _dk_half2[..., :_hd2] = -dense_k_assembled[..., _hd2:]
                                        _dk_half2[..., _hd2:] = dense_k_assembled[..., :_hd2]
                                        _cos_compute = _cos_d2.to(dense_k_assembled.dtype)
                                        _sin_compute = _sin_d2.to(dense_k_assembled.dtype)
                                        _rot = torch.empty_like(dense_k_assembled)
                                        torch.mul(dense_k_assembled, _cos_compute, out=_rot)
                                        _rot.addcmul_(_dk_half2, _sin_compute)
                                        _rot_state = {
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
                                                _half_suffix = _dk_half2[:, :, _s:_e]
                                                _half_suffix[..., :_hd2] = -_raw_suffix[..., _hd2:]
                                                _half_suffix[..., _hd2:] = _raw_suffix[..., :_hd2]
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
                                                    _raw_suffix,
                                                    _cos_compute[:, :, _s:_e],
                                                    out=_rot[:, :, _s:_e],
                                                )
                                                _rot[:, :, _s:_e].addcmul_(
                                                    _half_suffix,
                                                    _sin_compute[:, :, _s:_e],
                                                )
                                            _offset += _new_len
                                        _rot_state["lengths"] = _lengths

                                    _dk_combined = _rot
                                    _dv_combined = dense_v_assembled
                                # P1-6: Deferred batch dispatch — queue this session's call
                                # so we can dispatch all sessions in tight Python-free sequence.
                                if _triton_batch_enabled:
                                    _triton_batch_queue.append((
                                        b_idx,
                                        dict(
                                            q=query_states[b_idx:b_idx+1],
                                            block_indices=block_indices,
                                            pool=pool,
                                            dense_k=_dk_combined,
                                            dense_v=_dv_combined,
                                            num_key_value_groups=num_key_value_groups,
                                            R=kv_manager.rank,
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
                                    attn_out_b = native_triton_sparse_attn_decode_combined(
                                        q=query_states[b_idx:b_idx+1],
                                        block_indices=block_indices,
                                        pool=pool,
                                        dense_k=_dk_combined,
                                        dense_v=_dv_combined,
                                        num_key_value_groups=num_key_value_groups,
                                        R=kv_manager.rank,
                                        S_MAX=session_mbs,
                                        anchor_indices=anchor_indices,
                                        cos=cos_all,
                                        sin=sin_all,
                                        dense_len=dense_len,
                                    )
                            else:
                                attn_out_b = native_triton_sparse_attn_decode(
                                    q=query_states[b_idx:b_idx+1],
                                    block_indices=block_indices,
                                    pool=pool,
                                    dense_blocks=dense_blocks,
                                    active_k=dense_k_assembled,
                                    active_v=dense_v_assembled,
                                    num_key_value_groups=num_key_value_groups,
                                    R=kv_manager.rank,
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
                                            cos_dense = cos_all[0, dense_positions.clamp(max=cos_all.shape[1] - 1)].squeeze().unsqueeze(0).unsqueeze(1)
                                            sin_dense = sin_all[0, dense_positions.clamp(max=sin_all.shape[1] - 1)].squeeze().unsqueeze(0).unsqueeze(1)
                                            dense_k_rot = (dense_k_valid * cos_dense) + (rotate_half(dense_k_valid)) * sin_dense

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
                                        attn_out_full_approx = native_triton_sparse_attn_decode(
                                            q=query_states[b_idx:b_idx+1],
                                            block_indices=_full_bi,
                                            pool=pool,
                                            dense_blocks=_full_dn,
                                            active_k=dense_k_assembled,
                                            active_v=dense_v_assembled,
                                            num_key_value_groups=num_key_value_groups,
                                            R=kv_manager.rank,
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
                    attn_output = self.o_proj(attn_output)

                    outputs = (attn_output,)
                    if output_attentions:
                        outputs += (None,)
                    if use_cache:
                        outputs += (None,)
                    return outputs

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
                                    cos_all, sin_all = model.model.rotary_emb(value_states[b_idx:b_idx+1], hist_pos)

                                    if dense_k:
                                        k_dense = torch.cat(dense_k, dim=1).unsqueeze(0)
                                        v_dense = torch.cat(dense_v, dim=1).unsqueeze(0)

                                        dense_positions_tensor = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                        cos_dense = cos_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                        sin_dense = sin_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                        k_dense_rot = (k_dense * cos_dense) + (rotate_half(k_dense)) * sin_dense

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
                                    _thresh = float(os.environ.get("DIFFKV_MPS_IN_LOOP_EMPTY_CACHE_THRESHOLD_GB", "5.5")) * 1024 * 1024 * 1024
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
                        if os.environ.get("DIFFKV_CONTIGUOUS_PREFILL", "0") == "1":
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
                                kv_manager.capture_prefill_kv(
                                    sid, captured_layer_idx,
                                    unrot_key_states[b_idx:b_idx+1].detach(),
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
                                        unrot_key_states[b_idx:b_idx+1].detach(),
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
                                    chunk_unrot_k = unrot_key_states[b_idx:b_idx+1, :, c_start:c_end, :]

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
                                        cos_all, sin_all = model.model.rotary_emb(value_states[b_idx:b_idx+1], hist_pos)

                                        if dense_k:
                                            k_dense = torch.cat(dense_k, dim=1).unsqueeze(0)
                                            v_dense = torch.cat(dense_v, dim=1).unsqueeze(0)

                                            dense_positions_tensor = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                            cos_dense = cos_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                            sin_dense = sin_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1)
                                            k_dense_rot = (k_dense * cos_dense) + (rotate_half(k_dense)) * sin_dense

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
                                        _thresh = float(os.environ.get("DIFFKV_MPS_IN_LOOP_EMPTY_CACHE_THRESHOLD_GB", "5.5")) * 1024 * 1024 * 1024
                                        if torch.mps.driver_allocated_memory() > _thresh:
                                            torch.mps.empty_cache()

                                attn_outputs.append(torch.cat(chunk_outs, dim=2))
                            attn_output = torch.cat(attn_outputs, dim=0)

                    attn_weights = None

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                attn_output = self.o_proj(attn_output)

                if torch.isnan(attn_output).any():
                    print(f"[DiffKV DEBUG] NaN detected in attn_output! layer={captured_layer_idx}, q_len={q_len}, is_decode={is_decode}")
                    print(f"  query_states has nan: {torch.isnan(query_states).any().item()}")
                    print(f"  key_states has nan: {torch.isnan(key_states).any().item()}")
                    print(f"  value_states has nan: {torch.isnan(value_states).any().item()}")
                outputs = (attn_output,)
                if output_attentions:
                    outputs += (attn_weights,)
                if use_cache:
                    outputs += (None,)

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

            return diffkv_forward

        layer.self_attn._original_forward = layer.self_attn.forward
        layer.self_attn.forward = make_diffkv_forward(i).__get__(layer.self_attn, layer.self_attn.__class__)

    if hasattr(model, "lm_head"):
        original_lm_head_forward = model.lm_head.forward
        def last_token_lm_head_forward(hidden_states):
            if getattr(model, "_disable_lm_head_slicing", False):
                return original_lm_head_forward(hidden_states)
            if hidden_states.shape[1] > 1:
                return original_lm_head_forward(hidden_states[:, -1:, :])
            return original_lm_head_forward(hidden_states)
        model.lm_head.forward = last_token_lm_head_forward

    print("Differential KV Attention Interception Applied. [Phase 29: Zero-overhead decode active]")

"""
dkv_backend.py — DKV AttentionInterface backend for transformers 5.x

Overview
--------
transformers 5.x (≥5.0) ships a first-class custom-attention registry called
AttentionInterface.  Instead of monkey-patching every layer's .forward(), DKV
can register a named backend and let HF load the model with:

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="dkv"
    )

The registered function receives Q, K, V that are ALREADY:
  - projected (q_proj / k_proj / v_proj, or fused qkv_proj)
  - normalized  (q_norm / k_norm, e.g. Qwen3)
  - RoPE-rotated

This solves the three engineering questions simultaneously:
  Q1 — API port:        No more dkv_forward() touching HF internals.
  Q2 — Adaptive:        HF handles layer detection, fused QKV, QK-norm.
  Q3 — Lower hook:      The AttentionInterface IS the lower integration point.

Gates / env variables
---------------------
DKV_USE_ATTENTION_INTERFACE  (default "0")
    Set to "1" to activate this path.  Old monkey-patch path is used otherwise.
    Allows A/B testing with a single env flag.

DKV_INVERSE_ROPE             (default "1" when interface active)
    "1" → inverse-RoPE the incoming K before storing in the pool (correct).
    "0" → store the already-rotated K as-is (fast, slightly inaccurate at
          decode anchor re-rotation, useful for benchmarking the cost delta).

DKV_BENCHMARK_INVERSE_ROPE   (default "0")
    "1" → log per-layer wall-clock cost of the inverse-RoPE step.  Prints one
          line per layer per token during decode so you can measure the overhead.

The unrotated-K problem
-----------------------
DKV's compression stores UNROTATED K so it can re-apply RoPE at the correct
block anchor position during decode.  When using AttentionInterface, HF has
already applied RoPE before calling this function.

Three options were considered (see implementation plan):
  1. Inverse-RoPE on K using position_embeddings kwarg  ← chosen (Option 3)
  2. Store rotated K, change block format               ← possible future opt
  3. Request unrotated K from HF (needs HF PR)          ← long-term

Option 1 is implemented here, gated behind DKV_INVERSE_ROPE.

Inverse-RoPE identity
---------------------
Forward:  k_rot = k * cos + rotate_half(k) * sin
Inverse:  k     = k_rot * cos - rotate_half(k_rot) * sin
          (valid because RoPE is an orthogonal rotation, R^T = R^{-1})
"""

from __future__ import annotations

import os
import math
import time
import threading
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ── Gates ─────────────────────────────────────────────────────────────────────
_USE_ATTN_INTERFACE  = os.environ.get("DKV_USE_ATTENTION_INTERFACE", "0") == "1"
_INVERSE_ROPE        = os.environ.get("DKV_INVERSE_ROPE",            "1") == "1"
_BENCH_INVERSE_ROPE  = os.environ.get("DKV_BENCHMARK_INVERSE_ROPE",  "0") == "1"

# ── Module-level kv_manager / model registry ─────────────────────────────────
# Keyed by model object id so multiple models with different managers are safe.
_DKV_REGISTRY: dict[int, dict] = {}  # {model_obj_id: {"kv_manager": ..., "model_ref": ...}}
_REGISTRY_LOCK = threading.Lock()


def bind_kv_manager(model: "torch.nn.Module", kv_manager) -> None:
    """Associate a KVRuntimeManager with a loaded model.  Must be called after
    from_pretrained() when using the AttentionInterface path."""
    with _REGISTRY_LOCK:
        _DKV_REGISTRY[id(model)] = {"kv_manager": kv_manager, "model_ref": model}
    # Also stamp directly on the model for fast lookup inside the attn fn
    model._dkv_kv_manager = kv_manager


def _lookup_registry(module: "torch.nn.Module"):
    """Walk up the module hierarchy to find the registered kv_manager."""
    # Fast path: stamped directly during bind_kv_manager
    km = getattr(module, "_dkv_kv_manager", None)
    if km is not None:
        return km, module
    # Slow path: search registry by model ref walking parents
    # (Only needed if the user didn't call bind_kv_manager yet.)
    with _REGISTRY_LOCK:
        for entry in _DKV_REGISTRY.values():
            mref = entry["model_ref"]
            # Check if module is a descendant of this model
            try:
                for name, m in mref.named_modules():
                    if m is module:
                        return entry["kv_manager"], mref
            except Exception:
                pass
    return None, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotate-half: splits last dim in two halves and swaps them."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_inverse_rotary_pos_emb(
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse RoPE: given rotated key k_rot and the (cos, sin) used to rotate it,
    recover the original (unrotated) key k.

    Identity:  k = k_rot * cos - rotate_half(k_rot) * sin

    Proof:
        k_rot = k * cos + rotate_half(k) * sin
        Let R = [[cos, -sin], [sin, cos]] (block-diagonal).
        R^T = [[cos, sin], [-sin, cos]].
        R^T @ k_rot = R^T @ R @ k = I @ k = k.
        In element-wise form: k = k_rot * cos - rotate_half(k_rot) * sin  ✓

    Args
    ----
    k_rot : [B, Hkv, seq, D]  — rotated key from HF
    cos   : [1, seq, D] or [B, seq, D] or [B, 1, seq, D]  — same used by HF
    sin   : same shape as cos

    Returns
    -------
    k     : [B, Hkv, seq, D]  — unrotated key for DKV pool storage
    """
    t0 = time.perf_counter() if _BENCH_INVERSE_ROPE else 0.0

    # Normalise cos/sin shape → [B, 1, seq, D] to broadcast over Hkv
    if cos.dim() == 2:          # [seq, D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:        # [B, seq, D] or [1, seq, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    # dim==4: [B, 1, seq, D] — already correct

    k = k_rot * cos - rotate_half(k_rot) * sin

    if _BENCH_INVERSE_ROPE:
        if k.device.type == "cuda":
            torch.cuda.synchronize()
        elif k.device.type == "mps":
            torch.mps.synchronize()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        print(f"[DKV inverse_rope] shape={tuple(k_rot.shape)} "
              f"device={k_rot.device} elapsed={elapsed_us:.1f}µs", flush=True)

    return k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expansion: repeat key/value heads n_rep times."""
    if n_rep == 1:
        return hidden_states
    bs, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bs, num_kv_heads, n_rep, slen, head_dim
    )
    out = hidden_states.reshape(bs, num_kv_heads * n_rep, slen, head_dim)
    if hidden_states.device.type == "mps":
        return out  # MPS SDPA accepts non-contiguous views
    return out.contiguous()


def _dense_sdpa_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    num_kv_groups: int,
) -> Tuple[torch.Tensor, None]:
    """Pure dense SDPA — used when DKV bypass is active (short context)."""
    k_rep = repeat_kv(key, num_kv_groups)
    v_rep = repeat_kv(value, num_kv_groups)
    scale = scaling if scaling is not None else (1.0 / math.sqrt(query.shape[-1]))
    attn_out = F.scaled_dot_product_attention(
        query, k_rep, v_rep,
        attn_mask=attention_mask,
        scale=scale,
    )
    return attn_out, None


# ── Engage threshold (mirrors the one in dkv_attention.py) ────────────────────
def _get_engage_threshold() -> int:
    return int(os.environ.get("DKV_ENGAGE_THRESHOLD", "4096"))


def _should_bypass(kv_manager, session_ids, layer_idx, is_decode, q_len) -> bool:
    """Return True if this forward should fall back to pure dense SDPA."""
    if not is_decode:
        # Prefill: bypass on short prompts with no compressed history
        total_prompt_len = q_len
        primary_sid = session_ids[0] if session_ids else None
        if primary_sid and primary_sid != "dummy_session":
            if hasattr(kv_manager, "_session_token_ids"):
                toks = kv_manager._session_token_ids.get(primary_sid)
                if toks is not None:
                    total_prompt_len = max(total_prompt_len, toks.numel())
            if hasattr(kv_manager, "get_session_sequence_length"):
                total_prompt_len = max(
                    total_prompt_len,
                    kv_manager.get_session_sequence_length(primary_sid),
                )
        if total_prompt_len < _get_engage_threshold():
            for sid in session_ids:
                if sid != "dummy_session":
                    if hasattr(kv_manager, "get_streaming_blocks"):
                        if kv_manager.get_streaming_blocks(sid, 0):
                            return False  # has history → engage
            return True  # short context, no history → bypass
    else:
        # Decode: bypass if no compressed blocks exist yet
        primary_sid = session_ids[0] if session_ids else None
        if primary_sid and primary_sid != "dummy_session":
            if hasattr(kv_manager, "get_streaming_blocks"):
                return len(kv_manager.get_streaming_blocks(primary_sid, 0)) == 0
        return True
    return False


# ── Main AttentionInterface function ─────────────────────────────────────────

def dkv_attention_forward(
    module: "torch.nn.Module",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    DKV attention backend registered with transformers' AttentionInterface.

    Called by HF after:
      q_proj / k_proj / v_proj  (or fused qkv_proj)
      q_norm / k_norm            (Qwen3, etc.)
      RoPE

    DKV's job here is to:
      1. Recover unrotated K (inverse-RoPE) for pool storage.
      2. Route to sparse prefill or sparse decode kernel.
      3. Return (attn_output [B, H, seq, D], attn_weights or None).

    NOTE: o_proj is applied by HF AFTER this function returns.

    Gates
    -----
    DKV_INVERSE_ROPE=1   (default)  → inverse-RoPE on K before pool storage
    DKV_INVERSE_ROPE=0              → store rotated K as-is (fast, imprecise)
    DKV_BENCHMARK_INVERSE_ROPE=1   → print per-layer timing of inverse-RoPE
    """
    # ── Resolve kv_manager and model ref ─────────────────────────────────────
    # Fast path: injected as kwargs by functools.partial at registration time,
    # or can be looked up via registry.
    kv_manager = kwargs.get("_dkv_kv_manager") or _lookup_registry(module)[0]
    model_ref  = kwargs.get("_dkv_model_ref")  or _lookup_registry(module)[1]

    if kv_manager is None:
        # No manager bound — fall back to plain SDPA so the model still runs.
        bsz, num_heads, q_len, head_dim = query.shape
        num_kv_groups = num_heads // key.shape[1]
        return _dense_sdpa_fallback(query, key, value, attention_mask, scaling, num_kv_groups)

    # ── Layer metadata ────────────────────────────────────────────────────────
    layer_idx    = getattr(module, "layer_idx", 0) or 0
    bsz, num_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    num_kv_groups = num_heads // num_kv_heads
    session_ids  = getattr(model_ref, "_dkv_session_ids", ["default"] * bsz)
    is_decode    = (q_len == 1)

    # ── Bypass: short context / no compressed history ─────────────────────────
    # Gate: bypass decision is cheap and identical to the old patch.
    if _should_bypass(kv_manager, session_ids, layer_idx, is_decode, q_len):
        return _dense_sdpa_fallback(query, key, value, attention_mask, scaling, num_kv_groups)

    # ── Unrotated K recovery ──────────────────────────────────────────────────
    # DKV's pool stores UNROTATED K so it can re-apply RoPE at the correct
    # anchor position during decode.  HF has already applied RoPE before
    # calling this function, so we must invert it.
    #
    # Gate: DKV_INVERSE_ROPE
    #   "1" (default) → apply inverse-RoPE (correct semantics)
    #   "0"           → skip (store rotated K; useful for benchmarking the
    #                    cost of the inverse step itself)
    #
    # position_embeddings = (cos, sin) is passed by all v5 LLM models as a
    # kwarg.  We read it here; if absent we cannot invert (fall back to "0").
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = kwargs.get(
        "position_embeddings", None
    )

    if _INVERSE_ROPE and position_embeddings is not None:
        cos, sin = position_embeddings
        # DKV_BENCHMARK_INVERSE_ROPE logging happens inside apply_inverse_rotary_pos_emb
        unrot_key = apply_inverse_rotary_pos_emb(key, cos, sin)
    else:
        # Either gate is off, or position_embeddings not provided.
        # Store rotated K — decode accuracy will be slightly off at anchor
        # re-rotation, but prefill correctness is unaffected.
        if _INVERSE_ROPE and position_embeddings is None:
            # Warn once per layer
            _warn_once(
                layer_idx,
                f"[DKV] DKV_INVERSE_ROPE=1 but position_embeddings not passed "
                f"by model at layer {layer_idx}. Storing rotated K (less accurate). "
                f"Set DKV_INVERSE_ROPE=0 to suppress this warning."
            )
        unrot_key = key  # rotated K stored as fallback

    # Unrotated query for SRL routing (pre-RoPE, last token in prefill, all in decode)
    # For routing we need the un-rotated query too (same inverse).
    if _INVERSE_ROPE and position_embeddings is not None:
        if is_decode or q_len == 1:
            unrot_query = apply_inverse_rotary_pos_emb(query, cos, sin)
        else:
            # Only last token needed for SRL pre-warm during prefill
            unrot_query = apply_inverse_rotary_pos_emb(
                query[:, :, -1:, :], cos[:, -1:] if cos.dim() >= 3 else cos[-1:], sin[:, -1:] if sin.dim() >= 3 else sin[-1:]
            )
    else:
        unrot_query = query[:, :, -1:, :] if not is_decode else query

    # ── finalize_compressed_blocks (once per token at layer 0) ───────────────
    if layer_idx == 0 and hasattr(kv_manager, "finalize_compressed_blocks"):
        kv_manager.finalize_compressed_blocks()

    # ── SRL pre-warm: track last prefill query ────────────────────────────────
    if layer_idx == 0 and q_len > 1:
        if not hasattr(kv_manager, "_last_prefill_q"):
            kv_manager._last_prefill_q = {}
        for b_idx, sid in enumerate(session_ids):
            if sid != "dummy_session":
                kv_manager._last_prefill_q[sid] = unrot_query[b_idx, :, -1, :].clone().detach()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    # Import the extracted helpers from dkv_attention (avoids duplication).
    # These were already compiled / imported by the old path.
    from runtime.dkv_attention import (
        _dkv_decode_forward_impl,
        _dkv_prefill_forward_impl,
        _get_prefill_chunk_size,
        rotate_half as _rotate_half_ref,   # not used here, just sanity
    )

    if is_decode:
        attn_output = _dkv_decode_forward_impl(
            query=query,
            unrot_query=unrot_query,
            key=key,
            unrot_key=unrot_key,
            value=value,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            head_dim=head_dim,
            layer_idx=layer_idx,
            kv_manager=kv_manager,
            model_ref=model_ref,
            session_ids=session_ids,
            position_embeddings=position_embeddings,
        )
    else:
        attn_output = _dkv_prefill_forward_impl(
            query=query,
            unrot_query=unrot_query,
            key=key,
            unrot_key=unrot_key,
            value=value,
            attention_mask=attention_mask,
            scaling=scaling,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            head_dim=head_dim,
            layer_idx=layer_idx,
            kv_manager=kv_manager,
            model_ref=model_ref,
            session_ids=session_ids,
            position_embeddings=position_embeddings,
        )

    # ── NaN guard (mirrors old patch) ─────────────────────────────────────────
    if torch.isnan(attn_output).any():
        print(
            f"[DKV DEBUG] NaN in attn_output layer={layer_idx} q_len={q_len} "
            f"is_decode={is_decode} q_nan={torch.isnan(query).any().item()} "
            f"k_nan={torch.isnan(key).any().item()}",
            flush=True,
        )

    # AttentionInterface expects: (attn_output [B, H, seq, D], attn_weights or None)
    # o_proj is NOT applied here — HF applies it after this returns.
    return attn_output, None


# ── One-time warning dedup ────────────────────────────────────────────────────
_warned: set[int] = set()

def _warn_once(key: int, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(msg, flush=True)


# ── Registration entry point ──────────────────────────────────────────────────

def register_dkv_backend(kv_manager=None, model_ref=None) -> None:
    """
    Register DKV as a named attention backend in transformers' AttentionInterface.

    Call this BEFORE from_pretrained() (or before set_attn_implementation).

    Example usage
    -------------
    ::

        from runtime.dkv_backend import register_dkv_backend, bind_kv_manager

        register_dkv_backend()                # registers the backend name
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="dkv"
        )
        bind_kv_manager(model, kv_manager)    # connects KV manager to model

    Or with kv_manager at registration time (simpler, single-model use):
    ::

        register_dkv_backend(kv_manager=kv_manager)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="dkv"
        )
        bind_kv_manager(model, kv_manager)
    """
    import functools
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask
    except ImportError as e:
        raise RuntimeError(
            "DKV AttentionInterface backend requires transformers ≥ 5.0. "
            f"Import error: {e}"
        ) from e

    # Partial-bind kv_manager if provided at registration time.
    # Multi-model use: bind_kv_manager() per model is cleaner.
    if kv_manager is not None:
        forward_fn = functools.partial(
            dkv_attention_forward,
            _dkv_kv_manager=kv_manager,
            _dkv_model_ref=model_ref,
        )
    else:
        forward_fn = dkv_attention_forward

    AttentionInterface.register("dkv", forward_fn)

    # Register matching mask function — reuse sdpa's mask builder so causal,
    # padding, and sliding-window constraints are all handled correctly.
    # Without this, transformers skips mask creation and passes None.
    AttentionMaskInterface.register("dkv", sdpa_mask)

    gate_status = (
        f"DKV_INVERSE_ROPE={os.environ.get('DKV_INVERSE_ROPE','1')} "
        f"DKV_BENCHMARK_INVERSE_ROPE={os.environ.get('DKV_BENCHMARK_INVERSE_ROPE','0')}"
    )
    print(
        f"[DKV] Registered 'dkv' AttentionInterface backend. "
        f"Load model with attn_implementation='dkv'. [{gate_status}]",
        flush=True,
    )

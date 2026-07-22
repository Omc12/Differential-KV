"""
native_core/sparse_decode/simd_expand.py

Vectorized / tiled block expansion kernels for DKV.

Feature 3 of the k-transformers × DKV integration.

Provides:
  - CUDA/Triton: already handled directly in triton_fused_decode.py via autotune.
    This module exposes the INT8-V quantization helpers that the Triton path uses.

  - MLX (Apple Silicon): `mlx_block_expand_fast()` and `mlx_block_expand_fallback()`
    The fast path attempts `mx.fast.metal_kernel` (MLX >= 0.16) for a
    tile-parallel Metal shader. Falls back to plain mx.matmul chain if unavailable.

  - CPU: `cpu_block_expand()` — pure PyTorch, optionally with torch.compile.

All functions share the same interface:
  expand_kv(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)
  -> (K_reconstructed, V_reconstructed)

Env knobs:
  DKV_MLX_METAL_EXPAND   auto  # 1=force Metal, 0=force pure-MLX, auto=try Metal
  DKV_TRITON_INT8_V       0    # 1=quantize V_KV to int8 in Triton kernel
"""

from __future__ import annotations

import os
import math
from typing import Optional, Tuple

import torch


# ── Backend detection ─────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _has_metal_kernel() -> bool:
    """Check if mx.fast.metal_kernel is available (MLX >= 0.16)."""
    try:
        import mlx.core as mx
        return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
    except ImportError:
        return False


_METAL_EXPAND_ENV = os.environ.get("DKV_MLX_METAL_EXPAND", "auto").lower()
_INT8_V_ENV = os.environ.get("DKV_TRITON_INT8_V", "0") == "1"


# ── INT8 V Quantization Helpers (for Triton INT8_V path) ─────────────────────

def quantize_v_to_int8(
    V_fp16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize V [pool_size, RANK, kv_heads, head_dim] from fp16 → int8.

    Scale is per-row (per RANK component) across the spatial dimensions.
    Returns (V_int8, V_scale) where V_scale is [pool_size, RANK] float16.
    This is called once after pool writes, not in the hot decode path.

    Usage: gated by DKV_TRITON_INT8_V=1
    """
    orig_shape = V_fp16.shape          # [N, R, H, D]
    N, R = orig_shape[0], orig_shape[1]
    V_f = V_fp16.float().reshape(N, R, -1)   # [N, R, H*D]
    scale = V_f.abs().amax(dim=-1).clamp(min=1e-5) / 127.0   # [N, R]
    V_int8 = torch.clamp(torch.round(V_f / scale.unsqueeze(-1)), -127, 127).to(torch.int8)
    V_int8 = V_int8.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
    return V_int8, scale.to(torch.float16)


def dequantize_v_int8(
    V_int8: torch.Tensor,
    V_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize V_int8 [N, R, H, D] int8 back to fp16.
    V_scale: [N, R] fp16.
    """
    V_f = V_int8.float()
    scale = V_scale.float().unsqueeze(-1).unsqueeze(-1)   # [N, R, 1, 1]
    return (V_f * scale).to(torch.float16)


# ── CPU fallback: pure PyTorch block expansion ─────────────────────────────────

def cpu_block_expand(
    U_int8: torch.Tensor,       # [S, R] int8
    U_scale: torch.Tensor,      # scalar or [1] float
    V_K_fp16: torch.Tensor,     # [R, kv_heads, head_dim] fp16 — V basis for K
    V_V_fp16: torch.Tensor,     # [R, kv_heads, head_dim] fp16 — V basis for V
    anchor_K: torch.Tensor,     # [kv_heads, head_dim] fp16
    anchor_V: torch.Tensor,     # [kv_heads, head_dim] fp16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch block expansion: K = U @ V_K + anchor_K, V = U @ V_V + anchor_V.

    U_int8 is dequantized inline. Result shapes:
      K: [S+1, kv_heads, head_dim]  (+1 for anchor)
      V: [S+1, kv_heads, head_dim]

    This is the CPU fallback used when neither CUDA Triton nor MLX Metal is available.
    Also used as the reference implementation for correctness testing.
    """
    # Dequant U
    U_f = U_int8.float() * float(U_scale)           # [S, R]

    # V reshape: [R, kv_heads*head_dim]
    R = V_K_fp16.shape[0]
    HD = anchor_K.numel()
    V_K_flat = V_K_fp16.reshape(R, HD).float()      # [R, HD]
    V_V_flat = V_V_fp16.reshape(R, HD).float()      # [R, HD]

    kv_heads, head_dim = anchor_K.shape

    # Delta reconstruction
    delta_K = (U_f @ V_K_flat).to(torch.float16).reshape(-1, kv_heads, head_dim)   # [S, H, D]
    delta_V = (U_f @ V_V_flat).to(torch.float16).reshape(-1, kv_heads, head_dim)   # [S, H, D]

    # Prepend anchor
    K_out = torch.cat([anchor_K.unsqueeze(0), anchor_K.unsqueeze(0) + delta_K], dim=0)
    V_out = torch.cat([anchor_V.unsqueeze(0), anchor_V.unsqueeze(0) + delta_V], dim=0)
    return K_out, V_out


# ── MLX fallback: pure mx.matmul chain ─────────────────────────────────────────

def mlx_block_expand_fallback(
    U_int8,       # mx.array [S, R] int8
    U_scale,      # mx.array scalar float16
    V_K_fp16,     # mx.array [R, kv_heads, head_dim] float16
    V_V_fp16,     # mx.array [R, kv_heads, head_dim] float16
    anchor_K,     # mx.array [kv_heads, head_dim] float16
    anchor_V,     # mx.array [kv_heads, head_dim] float16
) -> Tuple:
    """
    Pure MLX block expansion without Metal kernel.
    Falls back to this if mx.fast.metal_kernel is unavailable.
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("MLX not installed; cannot use mlx_block_expand_fallback")

    # Dequant U: int8 → float16
    U_f = U_int8.astype(mx.float16) * U_scale      # [S, R]

    R = V_K_fp16.shape[0]
    kv_heads, head_dim = anchor_K.shape
    V_K_flat = V_K_fp16.reshape(R, -1)             # [R, H*D]
    V_V_flat = V_V_fp16.reshape(R, -1)             # [R, H*D]

    # Matrix multiply
    delta_K = (U_f @ V_K_flat).reshape(-1, kv_heads, head_dim)   # [S, H, D]
    delta_V = (U_f @ V_V_flat).reshape(-1, kv_heads, head_dim)   # [S, H, D]

    K_out = mx.concatenate([mx.expand_dims(anchor_K, 0), mx.expand_dims(anchor_K, 0) + delta_K], axis=0)
    V_out = mx.concatenate([mx.expand_dims(anchor_V, 0), mx.expand_dims(anchor_V, 0) + delta_V], axis=0)
    return K_out, V_out


# ── MLX fast path: mx.fast.metal_kernel tile ───────────────────────────────────

_METAL_KERNEL_AVAILABLE = False
_mlx_metal_expand_kernel = None


def _try_build_metal_kernel():
    """Attempt to build the mx.fast.metal_kernel tile. Cached after first call."""
    global _METAL_KERNEL_AVAILABLE, _mlx_metal_expand_kernel
    if _mlx_metal_expand_kernel is not None:
        return _METAL_KERNEL_AVAILABLE
    try:
        import mlx.core as mx
        if not (hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")):
            return False

        # Metal Shading Language source: tile-parallel U@V + anchor add
        # Threadgroup: (TILE_S=32, TILE_R=16). Each thread computes one output element.
        # Input layout:
        #   U_int8:  [S, R]         int8
        #   U_scale: [1]            float16
        #   V_flat:  [R, HD]        float16   (V_K or V_V flattened)
        #   anchor:  [HD]           float16
        # Output: [S+1, HD] float16
        _METAL_SOURCE = r"""
        #include <metal_stdlib>
        using namespace metal;

        kernel void dkv_block_expand(
            device const char*    U_int8  [[buffer(0)]],
            device const half*    U_scale [[buffer(1)]],
            device const half*    V_flat  [[buffer(2)]],
            device const half*    anchor  [[buffer(3)]],
            device half*          out     [[buffer(4)]],
            constant uint&        S       [[buffer(5)]],
            constant uint&        R       [[buffer(6)]],
            constant uint&        HD      [[buffer(7)]],
            uint2                 gid     [[thread_position_in_grid]]
        ) {
            // gid.x = token index (0 = anchor, 1..S = delta tokens)
            // gid.y = feature index in flattened output HD
            if (gid.x >= S + 1 || gid.y >= HD) return;

            half anc = anchor[gid.y];

            if (gid.x == 0) {
                // Anchor row: copy exactly
                out[gid.x * HD + gid.y] = anc;
                return;
            }

            uint s = gid.x - 1;   // delta token index
            float acc = 0.0f;
            float scale = float(U_scale[0]);
            for (uint r = 0; r < R; r++) {
                float u_val = float((int)U_int8[s * R + r]) * scale;
                acc += u_val * float(V_flat[r * HD + gid.y]);
            }
            out[gid.x * HD + gid.y] = half(acc) + anc;
        }
        """

        _mlx_metal_expand_kernel = mx.fast.metal_kernel(
            name="dkv_block_expand",
            input_names=["U_int8", "U_scale", "V_flat", "anchor"],
            output_names=["out"],
            source=_METAL_SOURCE,
        )
        _METAL_KERNEL_AVAILABLE = True
        return True
    except Exception as e:
        _METAL_KERNEL_AVAILABLE = False
        if os.environ.get("DKV_TELEMETRY", "0") == "1":
            print(f"[DKV simd_expand] Metal kernel build failed ({e}); using pure-MLX fallback")
        return False


def mlx_block_expand_fast(
    U_int8,
    U_scale,
    V_K_fp16,
    V_V_fp16,
    anchor_K,
    anchor_V,
) -> Tuple:
    """
    Fast MLX block expansion using mx.fast.metal_kernel when available.

    Tries to use the tile-parallel Metal shader; falls back to pure-MLX matmul
    if the kernel cannot be compiled (MLX version < 0.16 or unsupported device).

    See mlx_block_expand_fallback() for the pure-MLX reference implementation.
    """
    if _METAL_EXPAND_ENV == "0":
        return mlx_block_expand_fallback(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)

    if not _try_build_metal_kernel():
        return mlx_block_expand_fallback(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)

    try:
        import mlx.core as mx

        kv_heads, head_dim = anchor_K.shape
        R = V_K_fp16.shape[0]
        S = U_int8.shape[0]
        HD = kv_heads * head_dim

        V_K_flat = V_K_fp16.reshape(R, HD)       # [R, HD]
        V_V_flat = V_V_fp16.reshape(R, HD)       # [R, HD]
        anchor_K_flat = anchor_K.reshape(HD)
        anchor_V_flat = anchor_V.reshape(HD)

        grid = ((S + 1), HD, 1)
        threadgroup = (min(32, S + 1), min(16, HD), 1)

        K_flat = _mlx_metal_expand_kernel(
            inputs=[U_int8, U_scale, V_K_flat, anchor_K_flat],
            template=[("uint", S), ("uint", R), ("uint", HD)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(S + 1, HD)],
            output_dtypes=[mx.float16],
        )[0]

        V_flat = _mlx_metal_expand_kernel(
            inputs=[U_int8, U_scale, V_V_flat, anchor_V_flat],
            template=[("uint", S), ("uint", R), ("uint", HD)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(S + 1, HD)],
            output_dtypes=[mx.float16],
        )[0]

        K_out = K_flat.reshape(S + 1, kv_heads, head_dim)
        V_out = V_flat.reshape(S + 1, kv_heads, head_dim)
        return K_out, V_out

    except Exception as e:
        if os.environ.get("DKV_TELEMETRY", "0") == "1":
            print(f"[DKV simd_expand] Metal dispatch failed ({e}); falling back to pure-MLX")
        return mlx_block_expand_fallback(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)


# ── Unified dispatch ───────────────────────────────────────────────────────────

def expand_kv_mlx(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V) -> Tuple:
    """
    Dispatch to the best available MLX expansion path:
      1. mx.fast.metal_kernel tile (if DKV_MLX_METAL_EXPAND != 0 and MLX >= 0.16)
      2. pure-MLX matmul chain (fallback)
    """
    if _METAL_EXPAND_ENV == "1" or _METAL_EXPAND_ENV == "auto":
        return mlx_block_expand_fast(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)
    return mlx_block_expand_fallback(U_int8, U_scale, V_K_fp16, V_V_fp16, anchor_K, anchor_V)

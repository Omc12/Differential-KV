"""Production-Grade Group-Quantized Residual Buffers (INT4 / INT8).

Asymmetric Group Quantization for Differential-KV residual vectors:
- Bits: 4 (or 8)
- Group size: 64
- Memory reduction: 3.56x physical storage savings vs FP16
- Storage:
    packed_data: torch.int32 of shape [..., packed_width]
    scales:      torch.float16 of shape [..., num_groups]
    biases:      torch.float16 of shape [..., num_groups]
"""

import math
import torch
from typing import Tuple, Optional

# Cache constant shift tensors per device to avoid reallocations on the hot path
_SHIFTS_4BIT_CACHE = {}
_SHIFTS_8BIT_CACHE = {}


def _get_shifts(device: torch.device, bits: int) -> torch.Tensor:
    if bits == 4:
        s = _SHIFTS_4BIT_CACHE.get(device)
        if s is None:
            s = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=device, dtype=torch.int32)
            _SHIFTS_4BIT_CACHE[device] = s
        return s
    elif bits == 8:
        s = _SHIFTS_8BIT_CACHE.get(device)
        if s is None:
            s = torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int32)
            _SHIFTS_8BIT_CACHE[device] = s
        return s
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}. Supported: 4, 8")


def get_packed_width(head_dim: int, bits: int) -> int:
    elems_per_int = 32 // bits
    if head_dim % elems_per_int != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by {elems_per_int} for {bits}-bit packing"
        )
    return (head_dim * bits + 31) // 32


def quantize_residuals_group_asymmetric(
    x: torch.Tensor,
    group_size: int = 64,
    bits: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize tensor x [..., D] to packed int32 data, float16 scale, and float16 bias.

    Args:
        x: Input tensor with shape [..., head_dim]
        group_size: Group size for asymmetric quantization (default: 64)
        bits: Quantization bits (default: 4, or 8)

    Returns:
        (packed_q, scale, bias):
            packed_q: [..., packed_width] torch.int32
            scale:    [..., D // group_size] torch.float16
            bias:     [..., D // group_size] torch.float16
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    if D % group_size != 0:
        raise ValueError(f"head_dim {D} must be divisible by group_size {group_size}")
    
    elems_per_int = 32 // bits
    if D % elems_per_int != 0:
        raise ValueError(f"head_dim {D} must be divisible by {elems_per_int}")

    packed_width = (D * bits + 31) // 32
    num_groups = D // group_size
    qmax = (1 << bits) - 1

    # Reshape into groups
    flat_leading = math.prod(orig_shape[:-1])
    x_g = x.reshape(flat_leading, num_groups, group_size).float()

    mn = x_g.amin(dim=-1, keepdim=True)
    mx = x_g.amax(dim=-1, keepdim=True)
    scale = torch.clamp((mx - mn) / float(qmax), min=1e-7)

    # Quantize to integer
    q = torch.clamp(torch.round((x_g - mn) / scale), 0, qmax).to(torch.int32)
    q = q.reshape(flat_leading, D)

    # Pack into int32
    shifts = _get_shifts(x.device, bits)
    q_sub = q.reshape(flat_leading, packed_width, elems_per_int)
    packed_q = (q_sub << shifts).sum(dim=-1, dtype=torch.int32)

    # Format output shapes
    out_packed = packed_q.reshape(*orig_shape[:-1], packed_width)
    out_scale = scale.squeeze(-1).reshape(*orig_shape[:-1], num_groups).to(torch.float16)
    out_bias = mn.squeeze(-1).reshape(*orig_shape[:-1], num_groups).to(torch.float16)

    return out_packed, out_scale, out_bias


def dequantize_residuals_group_asymmetric(
    packed_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    group_size: int = 64,
    bits: int = 4,
    target_dtype: torch.dtype = torch.float16,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Dequantize packed int32 data, float16 scale, and float16 bias.

    Args:
        packed_q: [..., packed_width] torch.int32
        scale:    [..., num_groups] torch.float16
        bias:     [..., num_groups] torch.float16
        group_size: Group size (default 64)
        bits: Quantization bits (default 4 or 8)
        target_dtype: Desired output dtype (default float16)
        head_dim: Output head_dim (defaults to packed_width * 32 // bits)

    Returns:
        Tensor of shape [..., head_dim] in target_dtype
    """
    orig_shape = packed_q.shape
    packed_width = orig_shape[-1]
    elems_per_int = 32 // bits
    if head_dim is None:
        head_dim = packed_width * elems_per_int
    num_groups = head_dim // group_size
    mask = (1 << bits) - 1

    flat_leading = math.prod(orig_shape[:-1])
    p_flat = packed_q.reshape(flat_leading, packed_width)

    shifts = _get_shifts(packed_q.device, bits)
    unpacked = ((p_flat.unsqueeze(-1) >> shifts) & mask).reshape(
        flat_leading, num_groups, group_size
    )

    s_flat = scale.reshape(flat_leading, num_groups, 1).to(target_dtype)
    b_flat = bias.reshape(flat_leading, num_groups, 1).to(target_dtype)

    deq = (unpacked.to(target_dtype) * s_flat + b_flat).reshape(*orig_shape[:-1], head_dim)
    return deq

"""
kernels/adaptive_anchor_kernel.py

Fused kernel for adaptive anchor lookup and sparse KV retrieval.
Reduces host-device synchronization by performing anchor selection on-GPU.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def adaptive_anchor_lookup_kernel(
    Anchor_ptr, Indices_ptr, Out_ptr,
    stride_ab, stride_an, stride_ad,
    stride_ib, stride_in,
    stride_ob, stride_on, stride_od,
    n_anchors, d_model,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel: out[b, n, d] = anchors[b, indices[b, n], d]
    Optimized for GPU residency.
    """
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, d_model)
    
    # Load indices
    idx_ptrs = Indices_ptr + bid * stride_ib + offs_n * stride_in
    indices = tl.load(idx_ptrs, mask=offs_n < n_anchors, other=0)
    
    # Load anchors based on indices
    anchor_ptrs = Anchor_ptr + bid * stride_ab + indices[:, None] * stride_an + offs_d[None, :] * stride_ad
    anchors = tl.load(anchor_ptrs, mask=(offs_n[:, None] < n_anchors) & (offs_d[None, :] < d_model), other=0.0)
    
    # Store to output
    out_ptrs = Out_ptr + bid * stride_ob + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, anchors, mask=(offs_n[:, None] < n_anchors) & (offs_d[None, :] < d_model))

def fused_anchor_lookup(anchors, indices):
    B, N, D = anchors.shape
    _, num_idx = indices.shape
    out = torch.empty((B, num_idx, D), device=anchors.device, dtype=anchors.dtype)
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_idx, BLOCK_SIZE), B)
    
    adaptive_anchor_lookup_kernel[grid](
        anchors, indices, out,
        anchors.stride(0), anchors.stride(1), anchors.stride(2),
        indices.stride(0), indices.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_idx, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

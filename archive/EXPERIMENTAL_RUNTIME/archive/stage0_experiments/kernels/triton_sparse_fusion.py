"""
kernels/triton_sparse_fusion.py

Advanced Triton kernels for fused sparse attention and KV reconstruction.
Optimized for reduced warp divergence and maximized tensor-core utilization.
"""

import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def fused_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mn, stride_mk,
    stride_ob, stride_oh, stride_on, stride_od,
    n_heads, d_model, n_ctx,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused Sparse Attention: Only computes attention for masked positions.
    Reduces memory traffic by skipping sparse loads where mask is zero.
    """
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_ctx, BLOCK_SIZE_N)
    num_pid_n = tl.cdiv(n_ctx, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Batch and Head IDs
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    # Load mask chunk
    mask_ptrs = Mask_ptr + bid * stride_mb + hid * stride_mh + offs_m[:, None] * stride_mn + offs_n[None, :] * stride_mk
    mask = tl.load(mask_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_n[None, :] < n_ctx), other=0.0)
    
    # If entire block is masked out, skip
    if tl.sum(mask) == 0:
        return

    # Load Q chunk
    q_ptrs = Q_ptr + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_model), other=0.0)

    # Load K chunk
    k_ptrs = K_ptr + bid * stride_kb + hid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_model), other=0.0)

    # Compute dot product
    qk = tl.dot(q, tl.trans(k))
    qk = qk / tl.sqrt(d_model.to(tl.float32))
    
    # Apply mask
    qk = tl.where(mask > 0, qk, float("-inf"))
    
    # Softmax (simplified for block-level fusion)
    m_i = tl.max(qk, 1)
    p = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(p, 1)
    p = p / l_i[:, None]

    # Load V chunk
    v_ptrs = V_ptr + bid * stride_vb + hid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_model), other=0.0)

    # Aggregation
    out = tl.dot(p, v)

    # Store result
    out_ptrs = Out_ptr + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_model))

def triton_sparse_attention(q, k, v, mask):
    B, H, L, D = q.shape
    out = torch.empty_like(q)
    
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 64 # Assuming D is small enough or multiple of 64
    
    grid = (triton.cdiv(L, BLOCK_SIZE_N) * triton.cdiv(L, BLOCK_SIZE_N), B, H)
    
    fused_sparse_attention_kernel[grid](
        q, k, v, mask, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, D, L,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_D=BLOCK_SIZE_D,
        GROUP_SIZE_M=8
    )
    return out

"""
kernels/fused_retrieval_kernel.py

Fused retrieval kernel that merges anchor lookup and sparse KV fetching.
Eliminates redundant memory movement between retrieval passes.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def fused_retrieval_logic_kernel(
    Anchor_ptr, Sparse_KV_ptr, Routing_Table_ptr, Out_ptr,
    stride_ab, stride_an, stride_ad,
    stride_sb, stride_sn, stride_sd,
    stride_rtb, stride_rtn,
    stride_ob, stride_on, stride_od,
    n_tokens, d_model,
    BLOCK_SIZE: tl.constexpr
):
    """
    Combined: Out = Anchors[Routing] + SparseKV
    This kernel assumes Routing_Table maps output tokens to their corresponding anchor/sparse components.
    """
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, d_model)
    
    # Load Routing
    rt_ptrs = Routing_Table_ptr + bid * stride_rtb + offs_n * stride_rtn
    route_idx = tl.load(rt_ptrs, mask=offs_n < n_tokens, other=0)
    
    # Load Anchor
    anchor_ptrs = Anchor_ptr + bid * stride_ab + route_idx[:, None] * stride_an + offs_d[None, :] * stride_ad
    anchors = tl.load(anchor_ptrs, mask=(offs_n[:, None] < n_tokens) & (offs_d[None, :] < d_model), other=0.0)
    
    # Load Sparse KV
    sparse_ptrs = Sparse_KV_ptr + bid * stride_sb + offs_n[:, None] * stride_sn + offs_d[None, :] * stride_sd
    sparse_kv = tl.load(sparse_ptrs, mask=(offs_n[:, None] < n_tokens) & (offs_d[None, :] < d_model), other=0.0)
    
    # Fuse
    out = anchors + sparse_kv
    
    # Store
    out_ptrs = Out_ptr + bid * stride_ob + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=(offs_n[:, None] < n_tokens) & (offs_d[None, :] < d_model))

def fused_sparse_retrieval(anchors, sparse_kv, routing_table):
    B, _, D = anchors.shape
    _, N = routing_table.shape
    out = torch.empty((B, N, D), device=anchors.device, dtype=anchors.dtype)
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), B)
    
    fused_retrieval_logic_kernel[grid](
        anchors, sparse_kv, routing_table, out,
        anchors.stride(0), anchors.stride(1), anchors.stride(2),
        sparse_kv.stride(0), sparse_kv.stride(1), sparse_kv.stride(2),
        routing_table.stride(0), routing_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        N, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

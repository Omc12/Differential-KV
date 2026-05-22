"""
kernels/sparse_prefetch_kernel.py

Asynchronous sparse KV prefetching kernels.
Designed to overlap KV retrieval with computation phases.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def prefetch_sparse_kv_kernel(
    Source_ptr, Dest_ptr, Indices_ptr,
    stride_sb, stride_sn, stride_sd,
    stride_db, stride_dn, stride_dd,
    stride_ib, stride_in,
    n_indices, d_model,
    BLOCK_SIZE: tl.constexpr
):
    """
    Prefetches sparse KV pairs from source (main VRAM) to destination (fast cache).
    """
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, d_model)
    
    # Load Indices
    idx_ptrs = Indices_ptr + bid * stride_ib + offs_n * stride_in
    indices = tl.load(idx_ptrs, mask=offs_n < n_indices, other=0)
    
    # Load from Source
    src_ptrs = Source_ptr + bid * stride_sb + indices[:, None] * stride_sn + offs_d[None, :] * stride_sd
    data = tl.load(src_ptrs, mask=(offs_n[:, None] < n_indices) & (offs_d[None, :] < d_model), other=0.0)
    
    # Store to Dest
    dest_ptrs = Dest_ptr + bid * stride_db + offs_n[:, None] * stride_dn + offs_d[None, :] * stride_dd
    tl.store(dest_ptrs, data, mask=(offs_n[:, None] < n_indices) & (offs_d[None, :] < d_model))

def prefetch_sparse_kv(source, dest, indices):
    B, _, D = source.shape
    _, N = indices.shape
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), B)
    
    prefetch_sparse_kv_kernel[grid](
        source, dest, indices,
        source.stride(0), source.stride(1), source.stride(2),
        dest.stride(0), dest.stride(1), dest.stride(2),
        indices.stride(0), indices.stride(1),
        N, D,
        BLOCK_SIZE=BLOCK_SIZE
    )

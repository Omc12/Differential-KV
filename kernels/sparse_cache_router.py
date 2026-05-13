"""
kernels/sparse_cache_router.py

GPU-resident retrieval router for sparse KV cache.
Manages fast-path routing decisions directly on the device.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def cache_routing_kernel(
    Scores_ptr, Threshold_ptr, Route_ptr,
    stride_sb, stride_sn,
    stride_rb, stride_rn,
    n_tokens,
    BLOCK_SIZE: tl.constexpr
):
    """
    Decides which tokens go to fast-path (sparse) vs anchor-path.
    Calculates routing masks on GPU to avoid host roundtrips.
    """
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load Scores
    score_ptrs = Scores_ptr + bid * stride_sb + offs_n * stride_sn
    scores = tl.load(score_ptrs, mask=offs_n < n_tokens, other=-1.0)
    
    # Load Threshold (could be scalar or per-head)
    threshold = tl.load(Threshold_ptr)
    
    # Decide Route: 1 for sparse, 0 for anchor
    route = scores > threshold
    
    # Store Route
    route_ptrs = Route_ptr + bid * stride_rb + offs_n * stride_rn
    tl.store(route_ptrs, route.to(tl.int32), mask=offs_n < n_tokens)

def route_sparse_cache(scores, threshold):
    B, N = scores.shape
    routes = torch.empty((B, N), device=scores.device, dtype=torch.int32)
    
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor([threshold], device=scores.device, dtype=scores.dtype)
        
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE), B)
    
    cache_routing_kernel[grid](
        scores, threshold, routes,
        scores.stride(0), scores.stride(1),
        routes.stride(0), routes.stride(1),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return routes

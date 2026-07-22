"""
runtime/bandwidth_estimator.py

Detailed bandwidth analysis and arithmetic intensity calculation.
"""

import numpy as np
from typing import Dict, List, Tuple

def estimate_bandwidth_scaling(
    seq_len: int,
    heads: int,
    head_dim: int,
    rank: int,
    block_size: int,
    sparse_ratio: float = 0.01,
    dtype_bytes: int = 2
):
    feat_dim = 2 * heads * head_dim
    
    # 1. FP16 Traffic (Full cache read per decode step)
    # Traffic = seq_len * feat_dim * dtype_bytes
    fp16_traffic = seq_len * feat_dim * dtype_bytes
    
    # 2. DKV Traffic
    # Traffic = Anchor read (1 per block) + U read (rank per token) + V read (shared per block) + Sparse
    # For a decode step at seq_len, we need to read all past anchors and compressed deltas.
    num_blocks = seq_len // block_size
    
    anchor_traffic = num_blocks * feat_dim * dtype_bytes
    u_traffic = seq_len * rank * 2 # U is [1, rank] per token
    v_traffic = num_blocks * (rank * feat_dim * 4) # V is [rank, feat_dim] per block
    sparse_traffic = seq_len * (feat_dim * sparse_ratio * 6) # index + value
    
    dkv_traffic = anchor_traffic + u_traffic + v_traffic + sparse_traffic
    
    # 3. Arithmetic Intensity (FLOPs / Bytes)
    # Reconstruction FLOPs: U @ V is 2 * rank * feat_dim per token
    recon_flops = seq_len * (2 * rank * feat_dim)
    
    intensity = recon_flops / (dkv_traffic + 1e-9)
    
    return {
        "context": seq_len,
        "fp16_mb": fp16_traffic / (1024**2),
        "dkv_mb": dkv_traffic / (1024**2),
        "reduction_x": fp16_traffic / (dkv_traffic + 1e-9),
        "intensity": intensity
    }

def find_crossover(heads, head_dim, rank, block_size):
    """Find at what context length DKV becomes bandwidth-positive."""
    for ctx in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        stats = estimate_bandwidth_scaling(ctx, heads, head_dim, rank, block_size)
        if stats["reduction_x"] > 1.0:
            return ctx
    return -1

if __name__ == "__main__":
    H, D = 32, 128
    R = 16
    B = 64
    
    print(f"--- Bandwidth Scaling Analysis (H={H}, D={D}, R={R}, B={B}) ---")
    for ctx in [4096, 16384, 65536, 131072]:
        s = estimate_bandwidth_scaling(ctx, H, D, R, B)
        print(f"Context {ctx:>6}: FP16 {s['fp16_mb']:>8.1f}MB | DKV {s['dkv_mb']:>8.1f}MB | {s['reduction_x']:>5.2f}x reduction | Intensity: {s['intensity']:.2f}")
    
    crossover = find_crossover(H, D, R, B)
    print(f"\nCrossover Point: {crossover} tokens")

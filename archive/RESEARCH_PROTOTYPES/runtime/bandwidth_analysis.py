"""
runtime/bandwidth_analysis.py

Analyzes effective memory bandwidth for Differential KV.
"""

import numpy as np

def estimate_bandwidth(
    seq_len: int,
    heads: int,
    head_dim: int,
    rank: int,
    block_size: int,
    sparse_ratio: float = 0.01,
    dtype_bytes: int = 2 # FP16
):
    feat_dim = 2 * heads * head_dim
    
    # 1. Baseline FP16
    fp16_bytes_per_token = feat_dim * dtype_bytes
    
    # 2. INT8 KV
    int8_bytes_per_token = feat_dim * 1 + 4/block_size # 1 byte per element + scale
    
    # 3. DKV Low-Rank
    # Per block: 1 Anchor [feat_dim * 2] + (block_size-1) * (U [rank * 2] + V [rank * feat_dim * 4 / block_size])
    # Effective per token:
    anchor_share = (feat_dim * dtype_bytes) / block_size
    u_share = rank * 2 # U is [1, rank] per token
    v_share = (rank * feat_dim * 4) / block_size # V is shared across block
    dkv_lr_bytes = anchor_share + u_share + v_share
    
    # 4. DKV Low-Rank + Sparse
    sparse_elements = feat_dim * sparse_ratio
    sparse_share = sparse_elements * (4 + 2) # index (int32) + value (fp16)
    dkv_lrs_bytes = dkv_lr_bytes + sparse_share
    
    return {
        "fp16": fp16_bytes_per_token,
        "int8": int8_bytes_per_token,
        "dkv_lowrank": dkv_lr_bytes,
        "dkv_lowrank_sparse": dkv_lrs_bytes
    }

if __name__ == "__main__":
    # Example for Llama-7B head config
    H, D = 32, 128
    B = 64
    R = 16
    
    stats = estimate_bandwidth(8192, H, D, R, B)
    print("Memory Traffic per Token (Bytes):")
    for k, v in stats.items():
        print(f"  {k:<25}: {v:.1f} bytes ({(v/stats['fp16']*100):.1f}%)")

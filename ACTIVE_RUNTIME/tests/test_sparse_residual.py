import torch
import numpy as np
import sys
import os
import math
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.native_block_pool import NativeBlockPool
from native_core.compression.lowrank import compress_lowrank
from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode, fused_decode_mps

def test_sparse_residual_correctness():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Define dimensions
    num_kv_heads = 4
    head_dim = 64
    feat_dim = 2 * num_kv_heads * head_dim
    seq_len = 8
    rank = 4
    
    # We create a random base key/value delta
    torch.manual_seed(42)
    deltas = torch.randn(seq_len, feat_dim, device=device, dtype=torch.float32) * 0.1
    # The first token is the anchor token, so its delta must be exactly zero.
    deltas[0] = 0.0
    
    # Inject 4 orthogonal fact tokens.
    fact_indices = [1, 3, 5, 7]
    for idx in fact_indices:
        pattern = torch.zeros(feat_dim, device=device)
        start_dim = (idx * 32) % feat_dim
        pattern[start_dim : start_dim + 64] = 5.0
        deltas[idx] = pattern
    
    # 2. Compress using lowrank SVD with threshold=0.0 and frac=1.0 to capture ALL tokens as residuals
    lr_delta = compress_lowrank(deltas, rank=rank, error_threshold=0.0, max_residual_frac=1.0)
    
    k_positions = lr_delta.residual_K_positions.cpu().tolist()
    v_positions = lr_delta.residual_V_positions.cpu().tolist()
    print(f"Residual K positions: {k_positions}")
    print(f"Residual V positions: {v_positions}")
    
    # 3. Setup NativeBlockPool and write the block
    pool = NativeBlockPool(
        max_blocks=10,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rank=rank,
        max_seq_len=8,
        device=device,
        dtype=torch.float16,
        initial_blocks=2,
        num_layers=1,
        lazy=False,
    )
    
    pool_idx = pool.allocate_block()
    
    anchor_K = deltas[0, :num_kv_heads * head_dim].view(num_kv_heads, head_dim).to(torch.float16)
    anchor_V = deltas[0, num_kv_heads * head_dim:].view(num_kv_heads, head_dim).to(torch.float16)
    
    # Write the block with residuals
    pool.write_block(
        pool_idx=pool_idx,
        U=lr_delta.U,
        V=lr_delta.V,
        anchor_K=anchor_K,
        anchor_V=anchor_V,
        scale=lr_delta.scale,
        seq_len=seq_len,
        residual_K_positions=lr_delta.residual_K_positions,
        residual_K_values=lr_delta.residual_K_values,
        residual_V_positions=lr_delta.residual_V_positions,
        residual_V_values=lr_delta.residual_V_values
    )
    
    # 4. Setup Query and perform decode attention
    num_heads = num_kv_heads * 2 # 2 query heads per KV head (GQA factor = 2)
    Q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    
    # Let's run _pytorch_vectorized_sparse_attn_decode
    block_indices = torch.tensor([pool_idx], device=device, dtype=torch.long)
    blk_sizes = torch.tensor([seq_len], device=device, dtype=torch.int32)
    anchor_indices = torch.tensor([0], device=device, dtype=torch.long)
    
    out_with_res = _pytorch_vectorized_sparse_attn_decode(
        q=Q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=2,
        R=rank,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=None,
        sin=None,
        total_seq_len=seq_len,
    )
    
    # Let's temporarily disable residual in pool and run again to see the benefit
    orig_k_pos = pool.residual_K_positions.clone()
    orig_v_pos = pool.residual_V_positions.clone()
    
    pool.residual_K_positions[pool_idx] = -1
    pool.residual_V_positions[pool_idx] = -1
    
    out_no_res = _pytorch_vectorized_sparse_attn_decode(
        q=Q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=2,
        R=rank,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=None,
        sin=None,
        total_seq_len=seq_len,
    )
    
    # Restore pool positions
    pool.residual_K_positions.copy_(orig_k_pos)
    pool.residual_V_positions.copy_(orig_v_pos)
    
    # Check intermediate values
    print(f"out_with_res sum: {out_with_res.sum().item():.6f}")
    print(f"out_no_res sum: {out_no_res.sum().item():.6f}")
    print(f"Difference: {(out_with_res - out_no_res).abs().max().item():.6f}")
    
    # Compute the true dense attention output for comparison
    K_dense = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    V_dense = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    
    K_dense[0] = anchor_K.float()
    V_dense[0] = anchor_V.float()
    
    K_dense[1:] = anchor_K.float().unsqueeze(0) + deltas[1:, :num_kv_heads*head_dim].view(seq_len-1, num_kv_heads, head_dim)
    V_dense[1:] = anchor_V.float().unsqueeze(0) + deltas[1:, num_kv_heads*head_dim:].view(seq_len-1, num_kv_heads, head_dim)
    
    # Repeat KV for GQA
    K_dense_rep = K_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len, head_dim]
    V_dense_rep = V_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len, head_dim]
    
    Q_sq = Q.view(num_heads, head_dim).float()
    scores_dense = torch.bmm(Q_sq.unsqueeze(1), K_dense_rep.transpose(1, 2)).squeeze(1) / math.sqrt(head_dim) # [num_heads, seq_len]
    probs_dense = torch.softmax(scores_dense, dim=-1) # [num_heads, seq_len]
    
    # Now let's do the SAME duplicate anchor logic for a perfect mathematical comparison:
    # duplicate the anchor token in dense scores
    scores_anchor_dense = scores_dense[:, 0:1] # [num_heads, 1]
    scores_all_dense = torch.cat([scores_anchor_dense, scores_dense], dim=-1) # [num_heads, 1 + seq_len]
    probs_all_dense = torch.softmax(scores_all_dense, dim=-1) # [num_heads, 1 + seq_len]
    
    P_anchor_dense = probs_all_dense[:, 0]
    P_comp_dense = probs_all_dense[:, 1:]
    out_dense_dup = (P_anchor_dense.unsqueeze(-1) + P_comp_dense.sum(dim=1).unsqueeze(-1)) * anchor_V.float().repeat_interleave(2, dim=0)
    # Add delta contributions
    delta_V_dense = V_dense[1:] - anchor_V.float().unsqueeze(0) # [seq_len-1, num_kv_heads, head_dim]
    delta_V_dense_rep = delta_V_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len-1, head_dim]
    out_dense_dup = out_dense_dup + torch.bmm(P_comp_dense[:, 1:].unsqueeze(1), delta_V_dense_rep).squeeze(1)
    
    diff_with_res = (out_with_res.squeeze(0).squeeze(1).float() - out_dense_dup).abs().mean().item()
    diff_no_res = (out_no_res.squeeze(0).squeeze(1).float() - out_dense_dup).abs().mean().item()
    
    print(f"Mean error with sparse residual vs duplicate dense: {diff_with_res:.6f}")
    print(f"Mean error without sparse residual vs duplicate dense: {diff_no_res:.6f}")
    
    print(f"out_with_res[0, 0, 0, :10]: {out_with_res[0, 0, 0, :10].float().tolist()}")
    print(f"out_no_res[0, 0, 0, :10]: {out_no_res[0, 0, 0, :10].float().tolist()}")
    print(f"out_dense_dup[0, :10]: {out_dense_dup[0, :10].tolist()}")
    assert diff_with_res < 0.01, f"Attention output with residual correction error {diff_with_res} is too high"
    assert diff_with_res < diff_no_res, "Residual correction should improve accuracy compared to compression-only"

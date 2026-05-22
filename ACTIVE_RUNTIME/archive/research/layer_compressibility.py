"""
runtime/layer_compressibility.py

Phase 12 — Layer-by-Layer Compressibility Analysis

Analyzes individual transformer layers (or their FFN projections) to empirically
determine their ideal residency and compression strategies.

Measurements:
1. Routing Sparsity (Gate Activation Gini/Sparsity): 
   - Highly sparse layers can have lower FFN VRAM budgets (aggressive tiering).
   - Dense/uniform layers must remain fully resident.
2. Low-Rank Compressibility (SVD error on activation):
   - Smooth layers can use aggressive KV compression (rank-4).
   - High-variance layers (typically early/late layers) need dense KV or rank-32.

This provides the empirical mapping for Hybrid Dense/Sparse Layer Residency.
"""

import torch
import torch.nn.functional as F
from typing import Dict

def analyze_layer_ffn_compressibility(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    block_size: int = 128,
    num_samples: int = 100,
    seq_len: int = 64
) -> Dict[str, float]:
    """
    Empirically analyzes a layer's FFN to determine residency requirements.
    
    Returns:
      dict with sparsity score, rank-16 error, recommended_budget_ratio
    """
    hidden_dim = W_gate.shape[1]
    d_ff = W_gate.shape[0]
    total_blocks = d_ff // block_size
    
    # 1. Routing Sparsity Analysis
    # We probe the gate projection with Gaussian inputs to measure activation sparsity.
    # In a real run, this would use actual token embeddings.
    torch.manual_seed(42)
    x = torch.randn(num_samples, seq_len, hidden_dim, device=W_gate.device, dtype=W_gate.dtype) * 0.5
    
    # [num_samples, seq_len, d_ff]
    gate_full = F.linear(x, W_gate)
    gate_silu = F.silu(gate_full)
    
    # Block-wise importance
    # [num_samples, seq_len, total_blocks]
    gate_blocked = gate_silu.abs().view(num_samples, seq_len, total_blocks, block_size).mean(dim=-1)
    
    # Calculate how much activation mass is concentrated in the top-K blocks
    # We look at the top 30% of blocks.
    k_30 = max(1, int(total_blocks * 0.3))
    
    # Sort blocks by activation magnitude per token
    sorted_blocks, _ = torch.sort(gate_blocked, dim=-1, descending=True)
    
    # Mass in top 30% vs total mass
    top_mass = sorted_blocks[:, :, :k_30].sum(dim=-1)
    total_mass = sorted_blocks.sum(dim=-1) + 1e-9
    concentration_ratio = (top_mass / total_mass).mean().item()
    
    # 2. Output Compressibility (Low-rank approximation of the FFN output)
    # Does this layer's FFN produce low-rank updates?
    # FFN output: [num_samples * seq_len, hidden]
    with torch.no_grad():
        up_vals = F.linear(x, W_up)
        mixed = gate_silu * up_vals
        ffn_out = F.linear(mixed, W_down)
        
        # Flatten to [N, hidden]
        ffn_out_2d = ffn_out.view(-1, hidden_dim).float()
        
        # SVD on the FFN output
        # If the top 16 singular values explain most of the variance, the layer output is highly compressible.
        U, S, Vh = torch.linalg.svd(ffn_out_2d, full_matrices=False)
        
        top_16_variance = S[:16].pow(2).sum()
        total_variance = S.pow(2).sum()
        variance_explained_rank16 = (top_16_variance / total_variance).item()

    # Strategy Recommendation based on empirical data
    # If concentration is high, we don't need all blocks in VRAM (Aggressive FFN Tiering).
    if concentration_ratio > 0.8:
        recommended_budget = 0.3  # Only 30% budget needed
    elif concentration_ratio > 0.6:
        recommended_budget = 0.5
    else:
        recommended_budget = 1.0  # Keep dense

    # If variance explained is high, we can compress KV outputs heavily
    if variance_explained_rank16 > 0.9:
        recommended_kv_rank = 4
    elif variance_explained_rank16 > 0.7:
        recommended_kv_rank = 16
    else:
        recommended_kv_rank = 32 # dense or high-rank

    return {
        "routing_concentration_top30": round(concentration_ratio, 4),
        "svd_variance_explained_rank16": round(variance_explained_rank16, 4),
        "recommended_ffn_vram_budget": recommended_budget,
        "recommended_kv_rank": recommended_kv_rank
    }

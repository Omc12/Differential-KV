"""
runtime/geometric_token_pruning.py

Prunes tokens from the KV cache based on geometric importance.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

class GeometricTokenPruner:
    """
    Identifies and removes tokens that are not critical for manifold stability.
    """
    def __init__(self, retention_rate: float = 0.4):
        self.retention_rate = retention_rate

    def prune(
        self,
        k: torch.Tensor,                # [batch, n_heads, seq_len, head_dim]
        v: torch.Tensor,
        importance_scores: torch.Tensor # [batch, n_heads, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prunes KV cache based on importance.
        Returns (pruned_k, pruned_v, indices_kept)
        """
        B, H, S, D = k.shape
        k_keep = int(S * self.retention_rate)
        k_keep = max(1, k_keep)
        
        # Get indices of top importance tokens
        # We might want to keep the most recent tokens always
        # Combine importance with recency bias
        recency = torch.linspace(0, 1, S, device=k.device).view(1, 1, S)
        combined_score = 0.8 * importance_scores + 0.2 * recency
        
        _, indices = torch.topk(combined_score, k=k_keep, dim=-1, sorted=True)
        
        # Sort indices to maintain temporal order if possible (optional but usually better)
        indices, _ = torch.sort(indices, dim=-1)
        
        # Gather pruned KV
        # torch.gather is a bit tricky for multi-dim, we'll use a simpler approach for simulation
        pruned_k_list = []
        pruned_v_list = []
        
        for b in range(B):
            head_k = []
            head_v = []
            for h in range(H):
                idx = indices[b, h]
                head_k.append(k[b, h, idx])
                head_v.append(v[b, h, idx])
            pruned_k_list.append(torch.stack(head_k))
            pruned_v_list.append(torch.stack(head_v))
            
        pruned_k = torch.stack(pruned_k_list)
        pruned_v = torch.stack(pruned_v_list)
        
        return pruned_k, pruned_v, indices

if __name__ == "__main__":
    B, H, S, D = 1, 8, 1024, 64
    pruner = GeometricTokenPruner(retention_rate=0.3)
    
    k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    importance = torch.rand(B, H, S)
    
    pk, pv, idx = pruner.prune(k, v, importance)
    print(f"Pruning Complete. Original Size: {S}, Pruned Size: {pk.shape[2]}")
    print(f"Indices Kept: {idx[0, 0, :5]} ...")

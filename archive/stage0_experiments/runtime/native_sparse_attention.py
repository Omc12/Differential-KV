"""
runtime/native_sparse_attention.py

Implements sparse geometric token selection and execution.
Uses curvature-aware token prioritization for attention routing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class NativeSparseAttention(nn.Module):
    """
    Executes sparse attention using geometric token prioritization.
    """
    def __init__(self, head_dim: int, sparse_ratio: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.sparse_ratio = sparse_ratio
        self.scale = head_dim ** -0.5

    def select_geometric_tokens(
        self,
        query: torch.Tensor,     # [bsz, n_heads, 1, head_dim]
        keys: torch.Tensor,      # [bsz, n_heads, seq_len, head_dim]
        curvature: torch.Tensor, # [bsz, n_heads, seq_len]
        top_k: int
    ) -> torch.Tensor:
        """
        Selects tokens based on a combination of attention resonance and manifold curvature.
        """
        # 1. Base Resonance (Standard Dot Product)
        resonance = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        resonance = resonance.squeeze(-2) # [bsz, n_heads, seq_len]
        
        # 2. Curvature-aware Importance
        # Tokens in high-curvature regions are prioritized for stabilization
        importance = resonance + 0.5 * curvature
        
        # 3. Top-k selection
        _, indices = torch.topk(importance, top_k, dim=-1)
        return indices

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        curvature: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, n_heads, q_len, d = q.shape
        seq_len = k.shape[2]
        
        top_k = max(1, int(seq_len * self.sparse_ratio))
        
        # We assume q_len=1 for incremental decoding or use mean query for prefill
        if q_len > 1:
            q_ref = q.mean(dim=2, keepdim=True)
        else:
            q_ref = q
            
        indices = self.select_geometric_tokens(q_ref, k, curvature, top_k)
        
        # Gather sparse KV
        # indices: [bsz, n_heads, top_k]
        batch_idx = torch.arange(bsz, device=q.device).view(bsz, 1, 1)
        head_idx = torch.arange(n_heads, device=q.device).view(1, n_heads, 1)
        
        k_sparse = k[batch_idx, head_idx, indices, :] # [bsz, n_heads, top_k, d]
        v_sparse = v[batch_idx, head_idx, indices, :] # [bsz, n_heads, top_k, d]
        
        # Compute sparse attention
        attn_weights = torch.matmul(q, k_sparse.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        output = torch.matmul(attn_weights, v_sparse)
        return output

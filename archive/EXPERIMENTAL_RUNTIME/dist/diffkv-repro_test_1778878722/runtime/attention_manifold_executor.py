"""
runtime/attention_manifold_executor.py

Top-level executor for the NCAA attention architecture.
Coordinates geometric routing, sparse selection, and manifold tracking.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from runtime.geometric_attention_router import GeometricAttentionRouter
from runtime.native_sparse_attention import NativeSparseAttention
from runtime.geometric_flash_attention import GeometricFlashAttention

class AttentionManifoldExecutor(nn.Module):
    """
    Manages the execution flow of geometric attention across layers.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_heads"]
        self.head_dim = self.hidden_size // self.n_heads
        
        # Components
        self.router = GeometricAttentionRouter(self.hidden_size, self.n_heads)
        self.sparse_attn = NativeSparseAttention(self.head_dim, sparse_ratio=config.get("sparse_ratio", 0.1))
        self.flash_attn = GeometricFlashAttention(self.head_dim, self.n_heads)
        
        # State
        self.manifold_state = {
            "drift": 0.0,
            "curvature": 0.0,
            "entropy": 0.0
        }

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attractors: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes the most efficient attention path based on current manifold state.
        """
        # 1. Determine routing role
        q_mean = q.mean(dim=(1, 2)) # [bsz, hidden]
        geo_stats = torch.tensor([[self.manifold_state["drift"], 
                                   self.manifold_state["curvature"], 
                                   self.manifold_state["entropy"]]], 
                                 device=q.device)
        
        role_probs = self.router(q_mean, geo_stats)
        
        # 2. Selective Execution
        # If drift is high, we might use more stabilization heads
        # If complexity is low, we use more sparse heads
        
        if self.manifold_state["curvature"] > 1.5:
            # High curvature: Use full attention with stabilization bias
            output = self.flash_attn(q, k, v, attractors if attractors is not None else k[:, :, :1, :], mask)
        else:
            # Stable manifold: Use sparse geometric attention
            # (In a real implementation, we'd have pre-calculated curvature)
            sim_curvature = torch.zeros((q.shape[0], self.n_heads, k.shape[2]), device=q.device)
            output = self.sparse_attn(q, k, v, sim_curvature, mask)
            
        return output

    def update_stats(self, drift: float, curvature: float, entropy: float):
        """Updates the internal manifold state tracking."""
        self.manifold_state["drift"] = 0.9 * self.manifold_state["drift"] + 0.1 * drift
        self.manifold_state["curvature"] = 0.9 * self.manifold_state["curvature"] + 0.1 * curvature
        self.manifold_state["entropy"] = 0.9 * self.manifold_state["entropy"] + 0.1 * entropy

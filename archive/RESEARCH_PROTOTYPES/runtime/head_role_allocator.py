"""
runtime/head_role_allocator.py

Dynamic allocator for specialized head roles based on cognitive demand.
"""

import torch
import torch.nn as nn
from typing import Dict, List

class HeadRoleAllocator(nn.Module):
    """
    Decides which heads should perform which roles at any given timestep.
    """
    def __init__(self, n_heads: int, feat_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        
        # Policy network for role allocation
        self.policy_net = nn.Sequential(
            nn.Linear(feat_dim + 4, 128), # feat_dim + [drift, curvature, entropy, seq_len_norm]
            nn.ReLU(),
            nn.Linear(128, n_heads * 5) # 5 roles
        )

    def allocate(
        self, 
        context_embedding: torch.Tensor, # [batch, feat_dim]
        stability_metrics: torch.Tensor, # [batch, 4]
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Allocates roles to heads.
        """
        batch_size = context_embedding.shape[0]
        
        # Combine state information
        state = torch.cat([context_embedding, stability_metrics], dim=-1)
        
        # Predict role logits
        logits = self.policy_net(state).view(batch_size, self.n_heads, 5)
        
        # Gumbel-Softmax for differentiable hard allocation (or just softmax for soft)
        probs = torch.softmax(logits / temperature, dim=-1)
        
        roles = ["retrieval", "stabilization", "predictive", "resonance", "routing"]
        
        allocations = {}
        for i, role in enumerate(roles):
            allocations[role] = probs[:, :, i]
            
        return allocations

    def compute_specialization_entropy(self, allocations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measures how specialized the heads are (lower entropy = more specialized).
        """
        # allocations: {role: [B, H]}
        # Stack to [B, H, Roles]
        stacked = torch.stack(list(allocations.values()), dim=-1)
        entropy = -torch.sum(stacked * torch.log(stacked + 1e-9), dim=-1)
        return entropy.mean()

if __name__ == "__main__":
    B, H, D = 1, 32, 4096
    allocator = HeadRoleAllocator(H, D)
    
    ctx = torch.randn(B, D)
    metrics = torch.tensor([[0.05, 1.1, 0.7, 0.5]]) # drift, curvature, entropy, seq_len
    
    allocs = allocator.allocate(ctx, metrics)
    entropy = allocator.compute_specialization_entropy(allocs)
    
    print(f"Role Allocation Complete.")
    print(f"Specialization Entropy: {entropy.item():.4f}")
    for role, mask in allocs.items():
        print(f"Role: {role}, Top Head Mask: {mask[0, :5]}")

import torch
import torch.nn.functional as F
from typing import Optional

class AdaptiveAttentionDensity:
    """
    Dynamically adjusts attention sparsity (density) based on local context.
    Uses attention entropy and signal-to-noise ratio to prune KV pairs.
    """
    def __init__(self, target_density: float = 0.5, min_density: float = 0.1):
        self.target_density = target_density
        self.min_density = min_density

    def compute_mask(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Generates a sparse mask based on attention weight distributions.
        attn_weights: [B, H, Q_LEN, K_LEN]
        """
        # Compute entropy per head
        # Low entropy -> Sharp focus -> Can prune more aggressively (or keep sharp peaks)
        # High entropy -> Distributed focus -> Need higher density
        
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
        mean_entropy = entropy.mean(dim=-1, keepdim=True)
        
        # Heuristic: Density is proportional to normalized entropy
        # (This is a simplified version of adaptive density)
        density = torch.clamp(
            self.target_density * (mean_entropy / mean_entropy.max()), 
            min=self.min_density, 
            max=1.0
        )
        
        # Apply top-k masking based on density
        k = int(attn_weights.size(-1) * density.mean().item())
        _, indices = torch.topk(attn_weights, k, dim=-1)
        
        mask = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        
        return mask

    def apply_sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Fused-style sparse attention simulation.
        """
        attn = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1)**0.5)
        probs = F.softmax(attn, dim=-1)
        
        mask = self.compute_mask(probs)
        sparse_attn = probs * mask
        # Re-normalize
        sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-9)
        
        return torch.matmul(sparse_attn, v)

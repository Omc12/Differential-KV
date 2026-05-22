import torch
import torch.nn as nn

class TokenImportanceGeometry(nn.Module):
    """
    Estimates token importance using grounded geometric properties (L2 norms, etc.).
    Avoids 'manifold intelligence' narratives.
    """
    def __init__(self):
        super().__init__()

    def compute_importance(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        k, v: [batch, heads, seq_len, head_dim]
        Returns: [batch, heads, seq_len]
        """
        # Heuristic 1: Key Norm (higher norm often indicates higher attention focus)
        k_norm = torch.norm(k, p=2, dim=-1)
        
        # Heuristic 2: Value Norm (higher norm indicates higher information payload)
        v_norm = torch.norm(v, p=2, dim=-1)
        
        # Combine heuristics
        importance = k_norm * v_norm
        
        return importance

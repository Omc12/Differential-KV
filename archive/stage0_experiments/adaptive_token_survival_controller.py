import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class AdaptiveTokenSurvivalController:
    """
    Implements token importance scoring and survival filtering.
    Only high-value tokens remain active in the compute pipeline.
    """
    def __init__(self, target_ratio: float = 0.5, entropy_threshold: float = 0.8):
        self.target_ratio = target_ratio
        self.entropy_threshold = entropy_threshold

    def score_tokens(self, keys: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Scores tokens based on resonance and recency.
        """
        # keys: [bsz, n_heads, seq_len, d]
        # queries: [bsz, n_heads, 1, d]
        bsz, n_heads, seq_len, d = keys.shape
        
        # 1. Resonance (Attention-based importance)
        resonance = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(-2) # [bsz, n_heads, seq_len]
        importance = resonance.mean(dim=1) # [bsz, seq_len]
        
        # 2. Recency Bias (Most recent tokens are usually more important)
        recency = torch.linspace(0.5, 1.0, seq_len, device=keys.device).unsqueeze(0)
        
        scores = importance * recency
        return scores

    def filter_active_tokens(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of active tokens based on survival scores.
        """
        bsz, seq_len = scores.shape
        k = max(1, int(seq_len * self.target_ratio))
        
        _, indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        
        return mask

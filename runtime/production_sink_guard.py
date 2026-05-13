import torch
import torch.nn as nn
from typing import Optional, List

class ProductionSinkGuard(nn.Module):
    """
    Production-ready attention sink preservation.
    Ensures that critical tokens (sinks and anchors) are never evicted from KV cache.
    """
    def __init__(self, sink_size: int = 4, anchor_indices: Optional[List[int]] = None):
        super().__init__()
        self.sink_size = sink_size
        self.anchor_indices = anchor_indices or []
        
    def get_preservation_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask where True indicates a token must be preserved.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # Protect sinks (initial tokens)
        if seq_len > 0:
            mask[:min(seq_len, self.sink_size)] = True
            
        # Protect specific anchors
        for idx in self.anchor_indices:
            if idx < seq_len:
                mask[idx] = True
                
        return mask

    def apply_guard(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Protects sink and anchor tokens by inflating their importance scores.
        """
        seq_len = importance_scores.size(-1)
        mask = self.get_preservation_mask(seq_len, importance_scores.device)
        
        # Inflate importance of protected tokens to ensure they are never pruned
        # Use a large finite value to avoid NaN issues with some kernels
        importance_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), 1e9)
        
        return importance_scores

    def update_anchors(self, new_anchors: List[int]):
        """
        Dynamically update which indices are considered anchors.
        """
        self.anchor_indices = list(set(self.anchor_indices + new_anchors))

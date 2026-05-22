import torch
from typing import List, Optional

class FusedRetrievalMasker:
    """
    Orchestrates the fusion of pruning, retrieval, and sink masks.
    Designed to minimize memory traffic by generating bitmasks in-place.
    """
    def __init__(self, sink_tokens: int = 4, anchor_tokens: int = 16):
        self.sink_tokens = sink_tokens
        self.anchor_tokens = anchor_tokens

    def create_mask(
        self,
        seq_len: int,
        importance_scores: torch.Tensor,
        retrieval_indices: Optional[torch.Tensor] = None,
        density: float = 0.1
    ) -> torch.Tensor:
        """
        Creates a fused sparse mask based on scores and predefined anchors.
        """
        device = importance_scores.device
        batch_size, num_heads, _ = importance_scores.shape
        
        # Initialize with zeros
        mask = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.bool, device=device)
        
        # 1. Protect Sinks
        mask[:, :, :self.sink_tokens] = True
        
        # 2. Protect Retrieval Hotspots
        if retrieval_indices is not None:
            # Expand indices to match mask shape if necessary
            mask.scatter_(2, retrieval_indices, True)
            
        # 3. Dynamic Top-K Sparsity
        # Calculate how many more tokens we can keep
        k = int(seq_len * density) - self.sink_tokens
        if retrieval_indices is not None:
            k -= retrieval_indices.shape[-1]
        
        k = max(0, k)
        
        if k > 0:
            # Mask out already protected tokens to find top-k among the rest
            masked_scores = importance_scores.clone()
            masked_scores[:, :, :self.sink_tokens] = float("-inf")
            if retrieval_indices is not None:
                masked_scores.scatter_(2, retrieval_indices, float("-inf"))
                
            _, top_indices = torch.topk(masked_scores, k, dim=-1)
            mask.scatter_(2, top_indices, True)
            
        return mask

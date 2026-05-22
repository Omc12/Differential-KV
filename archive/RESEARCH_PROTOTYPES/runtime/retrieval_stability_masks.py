import torch
from typing import List, Optional

class RetrievalStabilityMasks:
    """
    Generates attention masks that stabilize retrieval paths.
    Ensures anchor tokens are always reachable by the query.
    """
    def __init__(self, anchor_count: int = 32):
        self.anchor_count = anchor_count

    def generate_mask(self, q_len: int, k_len: int, anchor_indices: torch.Tensor) -> torch.Tensor:
        """
        Generates a stability mask.
        anchor_indices: Indices of tokens identified as retrieval anchors.
        """
        # Base causal mask
        mask = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool), diagonal=1)
        
        # Protect anchors: set their mask values to False (unmasked)
        # We need to map the global anchor indices to the relative k_len indices.
        # Assuming k_len is the total available keys.
        
        # Stability logic: Any query should be able to attend to anchors
        # regardless of distance, even in sparse modes.
        stability_mask = torch.ones((q_len, k_len), dtype=torch.bool)
        
        # 1. Causal constraint (optional, depending on if we are in decoding)
        causal_mask = torch.arange(k_len).view(1, -1) <= torch.arange(q_len).view(-1, 1)
        
        # 2. Anchor protection
        anchor_mask = torch.zeros((q_len, k_len), dtype=torch.bool)
        anchor_mask[:, anchor_indices] = True
        
        # Final mask: Causal AND (SlidingWindow OR Anchor)
        # For simplicity, we just return the anchor-aware causal mask
        return causal_mask | anchor_mask

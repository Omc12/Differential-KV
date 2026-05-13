import torch
from typing import List, Dict

class RetrievalContentionPredictor:
    """
    Predicts cross-user contention based on current retrieval maps.
    Enables early sharding or scheduling adjustments.
    """
    def __init__(self, seq_len: int = 16384):
        self.usage_map = torch.zeros(seq_len)

    def update_usage(self, all_indices: torch.Tensor):
        """Updates the global usage heat map."""
        if all_indices.numel() == 0:
            return
        
        # Use atomic-like update on a shared map (mock)
        counts = torch.bincount(all_indices, minlength=self.usage_map.size(0))
        self.usage_map = self.usage_map * 0.9 + counts * 0.1

    def predict_contention(self, threshold: float = 2.0) -> torch.Tensor:
        """Returns indices predicted to be under high contention."""
        return torch.where(self.usage_map > threshold)[0]

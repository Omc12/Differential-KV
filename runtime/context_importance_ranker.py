import torch
import torch.nn as nn

class ContextImportanceRanker(nn.Module):
    """
    PHASE 7.5A: Context Importance Ranker
    Identifies high-information regions in the KV cache that require 
    denser anchor placement to maintain retrieval integrity.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.entropy_proj = nn.Linear(hidden_dim, 1)

    def rank_regions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes importance scores for context windows.
        Higher scores indicate regions that should be anchors.
        """
        # Information density approximation using state magnitude and projection
        # High magnitude often correlates with structural importance in LLMs
        magnitudes = torch.norm(hidden_states, dim=-1)
        projections = self.entropy_proj(hidden_states).squeeze(-1)
        
        importance = magnitudes * torch.sigmoid(projections)
        return importance

    def get_high_info_indices(self, importance: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        """
        Returns indices of regions exceeding the importance threshold.
        """
        return torch.where(importance > threshold)[0]

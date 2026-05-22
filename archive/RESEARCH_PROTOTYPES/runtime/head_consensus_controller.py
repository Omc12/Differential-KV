import torch
import torch.nn as nn

class HeadConsensusController(nn.Module):
    """
    Ensures that different attention heads aren't pruning wildly different sets of tokens.
    Promotes consensus on token importance to maintain structural stability.
    """
    def __init__(self, consensus_weight: float = 0.5):
        super().__init__()
        self.consensus_weight = consensus_weight

    def apply_consensus(self, head_importance_scores: torch.Tensor) -> torch.Tensor:
        """
        head_importance_scores: [batch, heads, seq_len]
        """
        # Calculate mean importance across all heads
        mean_importance = head_importance_scores.mean(dim=1, keepdim=True)
        
        # Pull individual head scores towards the mean
        consensus_scores = (1.0 - self.consensus_weight) * head_importance_scores + self.consensus_weight * mean_importance
        
        return consensus_scores

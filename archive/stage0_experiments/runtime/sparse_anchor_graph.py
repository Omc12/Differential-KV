import torch
from typing import Dict, List, Tuple

class SparseAnchorGraph:
    """
    Tracks dependencies between tokens using attention weights.
    Discovers 'hubs' that act as stable anchors for retrieval.
    """
    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.adj_matrix: Optional[torch.Tensor] = None
        self.node_importance: Optional[torch.Tensor] = None

    def update_graph(self, attn_weights: torch.Tensor):
        """
        Update the importance graph based on current attention weights.
        attn_weights: [H, Q, K]
        """
        # Collapse heads
        mean_attn = attn_weights.mean(dim=0) # [Q, K]
        
        # Incremental importance update
        k_len = mean_attn.size(-1)
        if self.node_importance is None or self.node_importance.size(0) != k_len:
            # Resize node importance if needed
            new_importance = torch.zeros(k_len, device=attn_weights.device)
            if self.node_importance is not None:
                new_importance[:self.node_importance.size(0)] = self.node_importance
            self.node_importance = new_importance
            
        # Accumulate attention received by each token
        # Tokens that are frequently attended to become anchors.
        self.node_importance = self.node_importance * self.decay + mean_attn.sum(dim=0)

    def get_anchors(self, top_k: int = 64) -> torch.Tensor:
        """Returns indices of the most important tokens."""
        if self.node_importance is None:
            return torch.tensor([], dtype=torch.long)
        _, indices = torch.topk(self.node_importance, min(top_k, self.node_importance.size(0)))
        return indices

import torch

class ContextImportanceRanker:
    """
    Ranks tokens by their importance to the global retrieval manifold.
    Uses attention centrality and gradient-like heuristics.
    """
    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.global_importance: Optional[torch.Tensor] = None

    def update_importance(self, attn_weights: torch.Tensor):
        """
        Updates global importance map based on incoming attention.
        attn_weights: [H, Q, K]
        """
        # Importance = Sum of attention received by each key
        received_attn = attn_weights.sum(dim=(0, 1)) # [K]
        
        if self.global_importance is None or self.global_importance.size(0) != received_attn.size(0):
            new_importance = torch.zeros_like(received_attn)
            if self.global_importance is not None:
                new_importance[:self.global_importance.size(0)] = self.global_importance
            self.global_importance = new_importance
            
        self.global_importance = self.global_importance * self.decay + received_attn

    def get_top_k_indices(self, k: int) -> torch.Tensor:
        """Returns the indices of the K most important tokens."""
        if self.global_importance is None:
            return torch.tensor([], dtype=torch.long)
        _, indices = torch.topk(self.global_importance, min(k, self.global_importance.size(0)))
        return indices

import torch

class SymbolicRelevanceTracker:
    """
    PHASE 20.1A: Tracks the relevance of symbolic spans over time.
    Prevents 'decay' of meaningful identifiers by reinforcing their presence.
    """
    def __init__(self, decay_rate: float = 0.95):
        self.decay_rate = decay_rate
        self.relevance_map = {} # {abs_index: relevance_score}

    def update_relevance(self, active_indices: torch.Tensor, attention_probs: torch.Tensor = None):
        """
        Updates relevance based on current attention access or mere presence.
        """
        # Decay existing scores
        for idx in self.relevance_map:
            self.relevance_map[idx] *= self.decay_rate
            
        # Boost active indices
        for idx in active_indices.flatten().tolist():
            self.relevance_map[idx] = self.relevance_map.get(idx, 0.0) + 1.0
            
        # Filter low relevance
        self.relevance_map = {k: v for k, v in self.relevance_map.items() if v > 0.01}

    def get_boost_factor(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns a boost multiplier for importance based on historical relevance.
        """
        boosts = torch.ones_like(indices, dtype=torch.float32)
        for i, idx in enumerate(indices.flatten().tolist()):
            boosts.flatten()[i] += self.relevance_map.get(idx, 0.0) * 0.1
        return boosts

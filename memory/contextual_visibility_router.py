import torch

class ContextualVisibilityRouter:
    """
    PHASE 20.1A: Routes symbolic signals based on contextual visibility.
    Decides whether a token should be in 'High Fidelity' (exact) or 'Sparse' (importance-weighted).
    """
    def __init__(self, fidelity_budget: int = 1024):
        self.fidelity_budget = fidelity_budget

    def route_tokens(self, importance_weights: torch.Tensor, salience_scores: torch.Tensor, window_size: int = 8):
        """
        Calculates routing probabilities with span-based windowing.
        """
        batch, q_len = importance_weights.shape
        combined_signal = importance_weights * torch.sigmoid(salience_scores)
        
        # Initial fidelity mask
        fidelity_mask = combined_signal > 0.8
        
        # Span Expansion (Protect neighboring tokens)
        expanded_mask = fidelity_mask.clone()
        for b in range(batch):
            indices = fidelity_mask[b].nonzero().flatten()
            for idx in indices:
                start = max(0, idx - window_size // 4)
                end = min(q_len, idx + window_size)
                expanded_mask[b, start:end] = True
        
        return expanded_mask, importance_weights

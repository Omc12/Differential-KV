import torch

class AdaptiveImportanceEstimator:
    """
    PHASE 20.1A: Refines raw salience into adaptive preservation weights.
    Balances symbolic visibility against cache budget constraints.
    """
    def __init__(self, target_budget: int = 1024):
        self.target_budget = target_budget
        self.dynamic_threshold = 1.5

    def calculate_importance(self, salience_scores: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """
        Adjusts importance based on noise and budget pressure.
        """
        # Noise-aware amplification
        # If noise is high, we boost the salience required for preservation
        effective_scores = salience_scores / (1.0 + noise_level)
        
        # Adaptive Thresholding (Simple EMA)
        current_mean = effective_scores.mean().item()
        self.dynamic_threshold = 0.9 * self.dynamic_threshold + 0.1 * (current_mean + 1.0)
        
        # Soft importance weights
        importance_weights = torch.sigmoid(effective_scores - self.dynamic_threshold)
        
        return importance_weights

    def get_preservation_mask(self, importance_weights: torch.Tensor, top_k: int = 512) -> torch.Tensor:
        """
        Hard mask based on the top-k most important tokens in the current chunk.
        """
        if importance_weights.shape[1] <= top_k:
            return importance_weights > 0.5
            
        values, indices = torch.topk(importance_weights, k=top_k, dim=-1)
        mask = torch.zeros_like(importance_weights, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask

import torch

class SoftPruningScheduler:
    """
    PHASE 19.0C: Soft Pruning Scheduler.
    Implements gradual pruning to avoid sudden relational collapse.
    """
    def __init__(self, decay_alpha: float = 0.1):
        self.decay_alpha = decay_alpha

    def schedule_pruning(self, importance_scores: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Adjusts importance scores to reflect a gradual decay rather than a 
        hard cutoff.
        """
        # Sort importance to find thresholds
        sorted_scores, _ = torch.sort(importance_scores, dim=-1)
        k = int(importance_scores.shape[1] * target_sparsity)
        threshold = sorted_scores[0, k]
        
        # Instead of 0/1 mask, we apply a sigmoid-like soft threshold
        # This keeps 'nearly important' tokens with some residual weight
        steepness = 10.0 / (threshold + 1e-6)
        soft_mask = torch.sigmoid(steepness * (importance_scores - threshold))
        
        return importance_scores * soft_mask

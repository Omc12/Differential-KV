import torch

class ContinuityGradientTracker:
    """
    PHASE 19.0C: Continuity Gradient Tracker.
    Monitors the 'smoothness' of attention gradients across the sparse KV cache.
    """
    def __init__(self):
        self.gradient_history = []

    def track_gradient(self, importance_scores: torch.Tensor) -> float:
        """
        Calculates the average absolute gradient of importance scores.
        High values indicate sharp cliffs.
        """
        if importance_scores.shape[1] < 2:
            return 0.0
            
        # Calculate finite differences
        diffs = torch.abs(importance_scores[:, 1:] - importance_scores[:, :-1])
        avg_gradient = diffs.mean().item()
        
        self.gradient_history.append(avg_gradient)
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
            
        return avg_gradient

    def get_cliff_metric(self) -> float:
        """
        Returns a metric indicating the presence of sharp discontinuities.
        """
        if not self.gradient_history:
            return 0.0
        return sum(self.gradient_history) / len(self.gradient_history)

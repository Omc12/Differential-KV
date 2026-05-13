import torch

class PruningJitterReducer:
    """
    Prevents 'flickering' where tokens are pruned and then immediately needed.
    Maintains a temporal 'cool-down' for pruning.
    """
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.importance_history = None

    def update_and_smooth(self, current_importance: torch.Tensor) -> torch.Tensor:
        """
        Smooths importance scores over time to reduce pruning jitter.
        """
        if self.importance_history is None:
            self.importance_history = current_importance
        else:
            # Simple alpha-beta smoothing or sliding window
            alpha = 0.8
            self.importance_history = alpha * self.importance_history + (1.0 - alpha) * current_importance
            
        return self.importance_history

    def reset(self):
        self.importance_history = None

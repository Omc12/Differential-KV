import torch

class AttentionPathCurvature:
    """
    Estimates the 'curvature' of attention paths to determine pruning safety.
    A high curvature path implies rapid changes in attention focus, making 
    pruning more risky.
    """

    def __init__(self):
        self.prev_importance = None

    def estimate_curvature(self, current_importance):
        """
        current_importance: [batch, seq_len]
        Calculates the first-order difference (velocity) and second-order 
        difference (acceleration/curvature) of token importance.
        """
        if self.prev_importance is None:
            self.prev_importance = current_importance
            return torch.zeros_like(current_importance)
            
        # Delta importance
        velocity = current_importance - self.prev_importance
        self.prev_importance = current_importance
        
        # In this context, we treat the 'curvature' as the magnitude of 
        # importance changes. Higher value = more volatile = less safe to prune.
        curvature = torch.abs(velocity)
        return curvature

if __name__ == "__main__":
    estimator = AttentionPathCurvature()
    imp1 = torch.tensor([[0.1, 0.2, 0.1]])
    imp2 = torch.tensor([[0.1, 0.8, 0.1]]) # Large change at index 1
    
    estimator.estimate_curvature(imp1)
    curvature = estimator.estimate_curvature(imp2)
    print(f"Curvature: {curvature.tolist()}")

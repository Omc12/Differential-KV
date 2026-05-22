import torch

class TokenGeometryEstimator:
    """
    Ranks token importance based on local geometry (e.g., L2 norm of KV states).
    Used ONLY for pruning safety estimation and anchor detection.
    """

    def __init__(self):
        pass

    def estimate_importance(self, k, v):
        """
        k, v: [batch, heads, seq_len, head_dim]
        Calculates the 'geometric energy' of each token.
        Higher energy tokens are often more important anchors.
        """
        # Calculate L2 norm across the head dimension
        # shape: [batch, heads, seq_len]
        k_norm = torch.norm(k, p=2, dim=-1)
        v_norm = torch.norm(v, p=2, dim=-1)
        
        # Combine and average across heads
        energy = (k_norm + v_norm) / 2.0
        importance = energy.mean(dim=1) # [batch, seq_len]
        
        return importance

    def detect_anchors(self, importance, top_k=4):
        """
        Returns indices of tokens with the highest geometric importance.
        """
        values, indices = torch.topk(importance, k=min(top_k, importance.shape[-1]), dim=-1)
        return indices

if __name__ == "__main__":
    estimator = TokenGeometryEstimator()
    k = torch.randn(1, 8, 10, 64)
    v = torch.randn(1, 8, 10, 64)
    # Make token 5 high energy
    k[:, :, 5, :] *= 10
    
    imp = estimator.estimate_importance(k, v)
    anchors = estimator.detect_anchors(imp, top_k=2)
    print(f"Top geometric anchors: {anchors.tolist()}")

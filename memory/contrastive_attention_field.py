import torch

class ContrastiveAttentionField:
    """PHASE 19.2B: Contrastive Attention Field"""
    def apply_contrast(self, scores: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        # Suppress noise locally
        scores[noise_mask] *= 0.1
        return scores

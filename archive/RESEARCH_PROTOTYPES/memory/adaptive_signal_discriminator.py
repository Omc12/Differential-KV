import torch

class AdaptiveSignalDiscriminator:
    """PHASE 19.2A: Adaptive Signal Discrimination"""
    def discriminate(self, scores: torch.Tensor, identity_mask: torch.Tensor) -> torch.Tensor:
        # Boost signals that have high identity uniqueness
        boost = torch.zeros_like(scores)
        boost[identity_mask] = 2000.0
        return scores + boost

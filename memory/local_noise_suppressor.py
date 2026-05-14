import torch

class LocalNoiseSuppressor:
    def detect_noise(self, scores: torch.Tensor, symbolic_mask: torch.Tensor) -> torch.Tensor:
        # High scores that are not symbolic are considered noise
        threshold = scores.mean() * 2.0
        return (scores > threshold) & (~symbolic_mask)

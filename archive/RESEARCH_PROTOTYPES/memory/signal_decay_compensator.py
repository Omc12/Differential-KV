import torch

class SignalDecayCompensator:
    def calculate_decay(self, energy: torch.Tensor) -> torch.Tensor:
        # returns a normalized decay factor
        max_e = energy.max()
        if max_e > 0:
            return 1.0 - (energy / max_e)
        return torch.zeros_like(energy)

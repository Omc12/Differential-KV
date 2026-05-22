import torch
import torch.nn as nn
from typing import Optional

class CognitiveEntropyEstimator:
    """
    Estimates cognitive entropy to guide synchronization and scaling policies.
    High entropy indicates unstable reasoning/desync.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        
    def estimate_entropy(self, manifold_states: torch.Tensor) -> torch.Tensor:
        """
        Calculates entropy based on the distribution of manifold activations.
        manifold_states: (batch, seq, d_model)
        """
        # Normalize states
        p = torch.softmax(manifold_states, dim=-1)
        
        # Shannon Entropy per step
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        
        # Average over sequence
        return entropy.mean()

    def get_sync_urgency(self, entropy: torch.Tensor) -> float:
        """
        Returns a value [0, 1] indicating how urgent a resync is.
        """
        # Base entropy for a 'stable' state might be around log(d_model) * 0.5
        # We normalize relative to theoretical max
        max_entropy = torch.log(torch.tensor(float(self.d_model)))
        urgency = (entropy / max_entropy).clamp(0, 1)
        return urgency.item()
        
    def detect_burst_requirement(self, entropy_trend: list) -> bool:
        """
        Detects if entropy is spiking rapidly, requiring a 'burst' of stabilization.
        """
        if len(entropy_trend) < 3:
            return False
        
        delta1 = entropy_trend[-1] - entropy_trend[-2]
        delta2 = entropy_trend[-2] - entropy_trend[-3]
        
        # Acceleration in entropy growth
        return delta1 > 0.1 and delta1 > delta2

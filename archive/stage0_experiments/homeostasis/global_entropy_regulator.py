import torch
import numpy as np
from typing import Dict, List

class GlobalEntropyRegulator:
    """
    Handles global entropy scaling and correction logic.
    """
    def __init__(self, d_model: int, target_entropy: float = 0.5):
        self.d_model = d_model
        self.target_entropy = target_entropy
        self.cooling_intensity = 0.0
        self.history = []
        self.drift_threshold = 0.15
        
    def measure_entropy(self, states: torch.Tensor) -> Dict:
        """
        Estimates Shannon entropy of the latent manifold states.
        """
        # Simplified latent entropy estimation using variance and distribution spread
        probs = torch.softmax(states, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        
        self.history.append(entropy)
        if len(self.history) > 1000:
            self.history.pop(0)
            
        return {
            "current_entropy": entropy,
            "target_entropy": self.target_entropy
        }
        
    def detect_drift(self, current_entropy: float) -> bool:
        """
        Detects if the entropy is drifting away from the stable equilibrium.
        """
        if len(self.history) < 10:
            return False
        
        baseline = np.mean(self.history[-10:-1])
        return abs(current_entropy - baseline) > self.drift_threshold
        
    def calculate_correction(self, current_entropy: float, pressure: float) -> float:
        """
        Calculates a correction factor (resonance boost) to suppress entropy.
        """
        error = current_entropy - self.target_entropy
        # Proportional-Derivative style correction
        correction = 1.0 + max(0, error * 2.0) + (pressure * 0.5)
        return min(correction, 5.0) # Cap at 5x resonance
        
    def apply_cooling(self, rate: float):
        """
        Reduces unnecessary stabilization energy.
        """
        self.cooling_intensity = min(1.0, self.cooling_intensity + rate)
        
    def estimate_long_horizon_drift(self, trajectory: List[float]) -> float:
        """
        Linear regression on entropy trajectory to predict long-term growth.
        """
        if len(trajectory) < 20:
            return 0.0
        
        x = np.arange(len(trajectory))
        y = np.array(trajectory)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

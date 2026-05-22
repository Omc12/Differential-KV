import torch
from typing import Dict, List
from evolution.cognitive_entropy_estimator import CognitiveEntropyEstimator

class SelfAdaptiveResonance:
    """
    Dynamically adjusts resonance intensity based on cognitive entropy.
    Implements low-energy coasting and burst stabilization.
    """
    def __init__(self, d_model: int):
        self.entropy_estimator = CognitiveEntropyEstimator(d_model)
        self.resonance_intensity = 1.0
        self.entropy_history = []
        self.min_intensity = 0.1
        self.max_intensity = 5.0
        
    def adjust_resonance(self, manifold_states: torch.Tensor) -> float:
        """
        Calculates new resonance intensity based on current entropy.
        """
        entropy = self.entropy_estimator.estimate_entropy(manifold_states)
        self.entropy_history.append(entropy.item())
        if len(self.entropy_history) > 100:
            self.entropy_history.pop(0)
            
        urgency = self.entropy_estimator.get_sync_urgency(entropy)
        is_burst = self.entropy_estimator.detect_burst_requirement(self.entropy_history)
        
        if is_burst:
            # Emergency stabilization
            self.resonance_intensity = self.max_intensity
        elif urgency < 0.2:
            # Low-energy coasting
            self.resonance_intensity = max(self.min_intensity, self.resonance_intensity * 0.9)
        else:
            # Normal adaptive adjustment
            target = 1.0 + (urgency * 2.0)
            self.resonance_intensity = 0.8 * self.resonance_intensity + 0.2 * target
            
        return self.resonance_intensity

    def get_policy_summary(self) -> Dict:
        mode = "COASTING" if self.resonance_intensity < 0.5 else "ACTIVE"
        if self.resonance_intensity > 3.0:
            mode = "BURST"
            
        return {
            "current_intensity": self.resonance_intensity,
            "mode": mode,
            "avg_entropy": sum(self.entropy_history) / len(self.entropy_history) if self.entropy_history else 0
        }

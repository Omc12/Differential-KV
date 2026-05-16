"""
analysis/resonance_energy.py
Phase 26: Cognitive Energy Minimization (CEM)
Defines the Cognitive Energy metric and tracks it over inference time.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional

class CognitiveEnergyModel:
    """
    Computes Cognitive Energy: E = drift + curvature + desync + entropy_instability
    """
    def __init__(self, drift_weight=1.0, curvature_weight=1.0, desync_weight=1.0, entropy_weight=1.0):
        self.weights = {
            "drift": drift_weight,
            "curvature": curvature_weight,
            "desync": desync_weight,
            "entropy_instability": entropy_weight
        }
        self.energy_history = []

    def compute_energy(self, 
                       drift: float, 
                       curvature: float, 
                       desync: float, 
                       entropy_instability: float) -> float:
        """
        E = w1*drift + w2*curvature + w3*desync + w4*entropy_instability
        """
        energy = (self.weights["drift"] * drift + 
                  self.weights["curvature"] * curvature + 
                  self.weights["desync"] * desync + 
                  self.weights["entropy_instability"] * entropy_instability)
        
        return float(energy)

    def record_energy(self, metrics: Dict[str, float]) -> float:
        energy = self.compute_energy(
            drift=metrics.get("hidden_drift", 0.0),
            curvature=metrics.get("trajectory_curvature", 0.0),
            desync=metrics.get("phase_desync", 0.0),
            entropy_instability=metrics.get("entropy_instability", 0.0)
        )
        self.energy_history.append(energy)
        return energy

    def get_history(self) -> List[float]:
        return self.energy_history

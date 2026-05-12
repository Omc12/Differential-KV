import torch
import numpy as np
from typing import Dict, List

class DriftTensorAnalyzer:
    """
    Builds and analyzes the DriftTensor[layer_i][layer_j].
    Tracks how manifold drift propagates across the network depth.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.drift_tensor = np.zeros((num_layers, num_layers))
        self.current_drifts = [0.0] * num_layers
        
    def update_drift(self, layer_idx: int, drift_value: float):
        """Updates the current drift value for a specific layer."""
        self.current_drifts[layer_idx] = drift_value
        
    def compute_drift_propagation(self):
        """
        Populates the DriftTensor based on current layer-wise drifts.
        DriftTensor[i][j] represents the relative drift coupling between layer i and j.
        """
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                # Drift coupling is the product of their individual drifts, 
                # normalized to show propagation relationship.
                self.drift_tensor[i, j] = self.current_drifts[i] * self.current_drifts[j]
                
    def get_drift_tensor(self) -> np.ndarray:
        return self.drift_tensor
    
    def analyze_bottlenecks(self) -> List[int]:
        """Identifies layers where drift propagation spikes."""
        propagation_magnitude = np.sum(self.drift_tensor, axis=0)
        # Layers with propagation above mean + std are bottlenecks
        mean = np.mean(propagation_magnitude)
        std = np.std(propagation_magnitude)
        
        bottlenecks = []
        for i, mag in enumerate(propagation_magnitude):
            if mag > mean + std:
                bottlenecks.append(i)
        return bottlenecks

    def estimate_collapse_probability(self) -> float:
        """Estimates the probability of global resonance collapse."""
        # Collapse is likely if the max drift coupling exceeds a critical threshold
        max_coupling = np.max(self.drift_tensor)
        # Sigmoid scaling for probability
        prob = 1.0 / (1.0 + np.exp(-5.0 * (max_coupling - 0.7)))
        return float(prob)

import torch
import numpy as np
from typing import List, Dict

class ResonanceDecayAnalysis:
    """
    Measures the decay of resonance signals over ultra-long horizons.
    """
    def __init__(self, horizon_length: int = 1000):
        self.horizon_length = horizon_length
        self.decay_curves = {}

    def measure_decay(self, layer_idx: int, signal_history: List[torch.Tensor]):
        """
        Calculates the autocorrelation decay of the resonance signal.
        """
        if len(signal_history) < 2:
            return 0.0
            
        signals = torch.stack(signal_history)
        base_signal = signals[0]
        
        correlations = []
        for i in range(len(signals)):
            corr = torch.nn.functional.cosine_similarity(
                base_signal.flatten(), signals[i].flatten(), dim=0
            ).item()
            correlations.append(corr)
            
        self.decay_curves[layer_idx] = correlations
        return correlations

    def get_half_life(self, layer_idx: int) -> int:
        if layer_idx not in self.decay_curves:
            return 0
            
        corrs = self.decay_curves[layer_idx]
        for i, c in enumerate(corrs):
            if c < 0.5:
                return i
        return len(corrs)

    def analyze_reinforcement_impact(self, 
                                     baseline_decay: List[float], 
                                     reinforced_decay: List[float]) -> float:
        """
        Calculates the 'survival gain' provided by recursive reinforcement.
        """
        baseline_area = np.trapz(baseline_decay)
        reinforced_area = np.trapz(reinforced_decay)
        
        if baseline_area == 0: return 0.0
        return (reinforced_area / baseline_area) - 1.0

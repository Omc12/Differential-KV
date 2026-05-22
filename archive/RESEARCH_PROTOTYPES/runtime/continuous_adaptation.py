"""
runtime/continuous_adaptation.py

Enables the Unified Cognitive Runtime to adapt its policies online.
Learns effective repair patterns and optimizes anchor placement.
"""

import numpy as np
from typing import List, Dict, Any, Optional

class ContinuousAdaptation:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learned_params = {
            "repair_threshold_offset": 0.0,
            "anchor_importance_bias": 0.0,
            "rank_scaling_multiplier": 1.0
        }
        self.performance_history: List[float] = []

    def observe_and_adapt(self, health_before: float, health_after: float, intervention_type: str):
        """
        Adjusts parameters based on the effectiveness of a repair intervention.
        """
        gain = health_after - health_before
        self.performance_history.append(gain)
        
        # Simple reinforcement signal
        learning_rate = 0.05
        if gain > 0:
            # Intervention worked, maybe lower the threshold to be more proactive
            self.learned_params["repair_threshold_offset"] -= learning_rate * gain
        else:
            # Intervention failed or was neutral, increase threshold or change strategy
            self.learned_params["repair_threshold_offset"] += learning_rate * abs(gain)
            
        # Clip offsets to reasonable ranges
        self.learned_params["repair_threshold_offset"] = np.clip(self.learned_params["repair_threshold_offset"], -0.2, 0.2)

    def get_adapted_threshold(self, base_threshold: float) -> float:
        return base_threshold + self.learned_params["repair_threshold_offset"]

    def optimize_anchor_policy(self, motif_frequency: Dict[str, int]):
        """
        Identifies persistent stable motifs and adjusts importance biases.
        """
        for motif, freq in motif_frequency.items():
            if freq > 10:
                # This motif is common and stable, give it a bias
                self.learned_params["anchor_importance_bias"] += 0.01

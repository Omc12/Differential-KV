"""
analysis/divergence_predictor.py
Phase 15: Divergence Prediction
Builds predictors for semantic/cognitive cliff onset and reasoning collapse.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional

class DivergencePredictor:
    def __init__(self, history_window: int = 5):
        self.history_window = history_window
        self.metric_history = []

    def update(self, metrics: Dict[str, float]):
        self.metric_history.append(metrics)
        if len(self.metric_history) > self.history_window:
            self.metric_history.pop(0)

    def predict_collapse(self) -> Dict[str, float]:
        """
        Returns probability of collapse based on trajectory trends.
        """
        if len(self.metric_history) < 2:
            return {"collapse_probability": 0.0, "reason": "insufficient_data"}

        curr = self.metric_history[-1]
        prev = self.metric_history[-2]

        # Indicators of impending collapse:
        # 1. Acceleration spike
        # 2. Curvature increase
        # 3. Fragmentation growth
        # 4. Stability decline
        
        accel_growth = max(0, curr["latent_acceleration"] - prev["latent_acceleration"])
        curvature_growth = max(0, curr["trajectory_curvature"] - prev["trajectory_curvature"])
        stability_drop = max(0, prev["cognitive_stability_score"] - curr["cognitive_stability_score"])
        frag_growth = max(0, curr["attention_fragmentation"] - prev["attention_fragmentation"])

        # Composite score
        risk_score = (accel_growth * 2.0) + (curvature_growth * 1.5) + (stability_drop * 3.0) + (frag_growth * 1.0)
        
        # Cliff detection: nonlinear jump in risk
        is_cliff = risk_score > 0.5
        
        collapse_prob = np.clip(risk_score, 0, 1)
        
        return {
            "collapse_probability": float(collapse_prob),
            "is_cliff_onset": bool(is_cliff),
            "primary_driver": self._identify_driver(accel_growth, curvature_growth, stability_drop, frag_growth)
        }

    def _identify_driver(self, accel, curv, stab, frag) -> str:
        drivers = {
            "acceleration": accel,
            "curvature": curv,
            "stability_loss": stab,
            "fragmentation": frag
        }
        return max(drivers, key=drivers.get)

if __name__ == "__main__":
    predictor = DivergencePredictor()
    # Mock sequence leading to collapse
    steps = [
        {"latent_acceleration": 0.01, "trajectory_curvature": 0.05, "cognitive_stability_score": 0.95, "attention_fragmentation": 0.1},
        {"latent_acceleration": 0.02, "trajectory_curvature": 0.06, "cognitive_stability_score": 0.93, "attention_fragmentation": 0.11},
        {"latent_acceleration": 0.15, "trajectory_curvature": 0.30, "cognitive_stability_score": 0.60, "attention_fragmentation": 0.25}, # Cliff
    ]
    
    for i, s in enumerate(steps):
        predictor.update(s)
        pred = predictor.predict_collapse()
        print(f"Step {i+1} Prediction: {pred}")

"""
analysis/death_spiral_analysis.py

Studies the 'Cognitive Death Spiral' - recursive repair failure and instability amplification.
Identifies signatures of irreversible collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

class DeathSpiralAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.intervention_history = []
        self.health_history = []
        self.drift_history = []

    def log_step(self, step: int, health: float, drift: float, intervention_type: Optional[str] = None):
        self.health_history.append(health)
        self.drift_history.append(drift)
        if intervention_type:
            self.intervention_history.append({"step": step, "type": intervention_type, "health": health})

    def analyze_spiral(self) -> Dict[str, Any]:
        """
        Analyzes the histories to detect a death spiral.
        """
        if len(self.intervention_history) < 3:
            return {"death_spiral_detected": False}

        # 1. Intervention Saturation
        # Frequent interventions without health improvement
        recent_interventions = [i for i in self.intervention_history if i["step"] > len(self.health_history) - 50]
        saturation_ratio = len(recent_interventions) / 50.0
        
        # 2. Repair Efficiency
        # Health gain per intervention
        improvements = []
        for i in range(1, len(self.intervention_history)):
            curr = self.intervention_history[i]
            prev = self.intervention_history[i-1]
            # Check health N steps after prev intervention
            # (Simplified: just compare health at intervention times)
            improvements.append(curr["health"] - prev["health"])
            
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # 3. Recursive Instability
        # Does drift increase despite repairs?
        drift_trend = np.polyfit(np.arange(len(self.drift_history)), self.drift_history, 1)[0] if len(self.drift_history) > 10 else 0
        
        # 4. Death Spiral Signature
        # Saturation is high, improvement is low/negative, drift is increasing
        is_spiral = saturation_ratio > 0.4 and avg_improvement < 0.05 and drift_trend > 0
        
        # Taxonomy of collapse
        taxonomy = "stable"
        if is_spiral:
            taxonomy = "irreversible_death_spiral"
        elif saturation_ratio > 0.4:
            taxonomy = "intervention_saturation"
        elif avg_improvement < 0:
            taxonomy = "harmful_intervention"
        elif self.health_history[-1] < 0.2:
            taxonomy = "catastrophic_collapse"

        return {
            "death_spiral_detected": bool(is_spiral),
            "saturation_ratio": float(saturation_ratio),
            "avg_repair_efficiency": float(avg_improvement),
            "drift_amplification": float(drift_trend),
            "collapse_taxonomy": taxonomy,
            "irreversibility_confidence": float(1.0 if is_spiral and self.health_history[-1] < 0.1 else 0.5)
        }

    def generate_saturation_curve(self) -> Dict[str, List[float]]:
        """
        Returns data for repair saturation plot (Intervention Count vs Health).
        """
        counts = []
        healths = []
        running_count = 0
        for i, h in enumerate(self.health_history):
            if any(intv["step"] == i for intv in self.intervention_history):
                running_count += 1
            counts.append(running_count)
            healths.append(h)
            
        return {"intervention_counts": counts, "health_scores": healths}

"""
analysis/attractor_energy_metrics.py
Phase 26: Cognitive Energy Minimization (CEM)
Measures stabilization costs and identifies high-cost regions.
"""

import numpy as np
from typing import List, Dict

class AttractorEnergyMetrics:
    def __init__(self):
        self.stabilization_costs = []
        self.high_cost_regions = []

    def record_step(self, step: int, cost: float, energy: float):
        self.stabilization_costs.append(cost)
        if cost > 0.5: # Empirical threshold for "high-cost"
            self.high_cost_regions.append({"step": step, "cost": cost, "energy": energy})

    def get_average_cost(self) -> float:
        return float(np.mean(self.stabilization_costs)) if self.stabilization_costs else 0.0

    def get_efficiency_ratio(self, coherence: float) -> float:
        # Higher is better
        total_cost = sum(self.stabilization_costs)
        return coherence / (total_cost + 1e-9)

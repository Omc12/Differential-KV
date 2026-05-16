"""
analysis/energy_landscape.py
Phase 26: Cognitive Energy Minimization (CEM)
Maps latent trajectories into energy basins and computes transition probabilities.
"""

import numpy as np
from typing import List, Dict, Tuple

class EnergyLandscapeMapper:
    def __init__(self, stable_threshold: float = 0.1, collapse_threshold: float = 0.5):
        self.stable_threshold = stable_threshold
        self.collapse_threshold = collapse_threshold
        self.transitions = {
            "stable": {"stable": 0, "semi-stable": 0, "collapse": 0},
            "semi-stable": {"stable": 0, "semi-stable": 0, "collapse": 0},
            "collapse": {"stable": 0, "semi-stable": 0, "collapse": 0}
        }
        self.total_counts = {"stable": 0, "semi-stable": 0, "collapse": 0}
        self.current_basin = "stable"

    def classify_basin(self, energy: float) -> str:
        if energy < self.stable_threshold:
            return "stable"
        elif energy < self.collapse_threshold:
            return "semi-stable"
        else:
            return "collapse"

    def update_trajectory(self, energy: float):
        new_basin = self.classify_basin(energy)
        self.transitions[self.current_basin][new_basin] += 1
        self.total_counts[new_basin] += 1
        self.current_basin = new_basin

    def get_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        probs = {}
        for source, targets in self.transitions.items():
            total = sum(targets.values())
            probs[source] = {t: (v / total if total > 0 else 0.0) for t, v in targets.items()}
        return probs

    def get_basin_stats(self) -> Dict[str, int]:
        return self.total_counts

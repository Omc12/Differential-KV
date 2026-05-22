"""
economy/attractor_propagation_engine.py

Policy-driven engine for propagating stable attractors through the collective.
"""

import torch
from typing import Dict, List, Optional, Any

class AttractorPropagationEngine:
    """
    Determines which attractors should be broadcast to the entire federation.
    Implements propagation policies (e.g., 'propagate if stability > threshold').
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.propagation_threshold = config.get("propagation_threshold", 0.95)
        self.propagation_history = []

    def evaluate_for_propagation(self, attractor_id: str, metrics: Dict[str, float]) -> bool:
        """
        Decides if an attractor should be propagated based on its metrics.
        """
        stability = metrics.get("stability", 0.0)
        generality = metrics.get("generality", 0.0) # How many agents benefit
        
        score = (stability + generality) / 2.0
        
        if score > self.propagation_threshold:
            self.propagation_history.append({"id": attractor_id, "score": score, "time": "now"})
            return True
        return False

    def get_propagation_policies(self) -> List[str]:
        """Returns the currently active propagation policies."""
        return ["high_stability_broadcast", "consensus_driven_sharing"]

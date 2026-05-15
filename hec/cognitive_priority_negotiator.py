
import torch
from typing import Dict, Any, List

class CognitivePriorityNegotiator:
    """
    PHASE 22.3: HEC - Cognitive Priority Negotiator.
    Arbitrates conflicts between specialized modes over compute priority.
    """
    def __init__(self):
        self.priority_stack: List[str] = []
        self.arbitration_stability = 1.0
        self.conflict_history = []

    def negotiate_priority(self, 
                           mode_priorities: Dict[str, float]) -> Dict[str, float]:
        """
        Balances competing priorities to avoid starvation.
        """
        if not mode_priorities:
            return {}
            
        # Detect conflicts (multiple modes with high priority)
        high_priorities = {k: v for k, v in mode_priorities.items() if v > 0.7}
        
        if len(high_priorities) > 1:
            self.conflict_history.append(list(high_priorities.keys()))
            # Arbitration: slightly dampen all high priorities to share budget
            # This prevents one mode from hogging the entire runtime
            dampening = 0.85
            for mode in high_priorities:
                mode_priorities[mode] *= dampening
            
            self.arbitration_stability *= 0.99 # Slight decay on conflict
        else:
            self.arbitration_stability = min(1.0, self.arbitration_stability + 0.01)
            
        return mode_priorities

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "arbitration_stability": self.arbitration_stability,
            "conflict_intensity": len(self.conflict_history)
        }

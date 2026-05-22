from typing import Dict, List
import numpy as np

class EvolutionConstraintEngine:
    """
    Ensures bounded policy evolution to prevent chaotic self-modification.
    """
    def __init__(self, max_change_rate: float = 0.2):
        self.max_change_rate = max_change_rate
        self.last_policy = None
        self.drift_history = []
        
    def check_bounds(self, proposed_policy: Dict) -> bool:
        """
        Calculates the delta from the last known stable policy.
        """
        if self.last_policy is None:
            self.last_policy = proposed_policy
            return True
            
        # Calculate policy drift (simplified)
        drift = 0.0
        keys = set(proposed_policy.keys()) & set(self.last_policy.keys())
        for k in keys:
            if isinstance(proposed_policy[k], (int, float)):
                drift += abs(proposed_policy[k] - self.last_policy[k])
        
        self.drift_history.append(drift)
        if drift > self.max_change_rate:
            return False
            
        self.last_policy = proposed_policy
        return True
        
    def get_stability_score(self) -> float:
        if not self.drift_history:
            return 1.0
        return float(1.0 - np.mean(self.drift_history))

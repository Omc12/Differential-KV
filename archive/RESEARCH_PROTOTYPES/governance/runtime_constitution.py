from typing import Dict

class RuntimeConstitution:
    """
    Hardcoded safety limits and invariants for the cognitive runtime.
    """
    def __init__(self):
        self.limits = {
            "max_resonance": 10.0,
            "min_resonance": 0.05,
            "max_manifold_growth": 1.5, # Factor per 1000 steps
            "min_synchronization": 0.1,
            "max_entropy_budget": 2.0
        }
        self.violation_count = 0
        
    def check(self, policy: Dict) -> bool:
        """
        Validates a policy against the constitution.
        """
        for key, value in policy.items():
            if key == "resonance_intensity":
                if value > self.limits['max_resonance'] or value < self.limits['min_resonance']:
                    self.violation_count += 1
                    return False
            if key == "entropy":
                if value > self.limits['max_entropy_budget']:
                    self.violation_count += 1
                    return False
        return True
        
    def get_default_safe_policy(self) -> Dict:
        return {
            "resonance_intensity": 1.0,
            "manifold_expansion": 1.0,
            "sync_damping": 0.5,
            "mode": "SAFE_MODE"
        }

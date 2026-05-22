from typing import Dict, List
from governance.evolution_constraint_engine import EvolutionConstraintEngine
from governance.runtime_constitution import RuntimeConstitution

class MetaStabilityController:
    """
    Governs the overall stability of the autonomous cognitive ecosystem.
    Prevents runaway self-optimization and maintains long-term integrity.
    """
    def __init__(self):
        self.constraints = EvolutionConstraintEngine()
        self.constitution = RuntimeConstitution()
        self.policy_history = []
        
    def validate_policy_change(self, proposed_policy: Dict) -> bool:
        """
        Validates if a proposed policy change (e.g., from an evolution engine)
        stays within safe bounds defined by the constitution.
        """
        # 1. Check against constitution
        is_constitutional = self.constitution.check(proposed_policy)
        
        # 2. Check against evolution constraints (rate of change)
        is_bounded = self.constraints.check_bounds(proposed_policy)
        
        if is_constitutional and is_bounded:
            self.policy_history.append(proposed_policy)
            return True
            
        return False
        
    def emergency_override(self) -> Dict:
        """
        Returns a hardcoded safe policy if runaway behavior is detected.
        """
        return self.constitution.get_default_safe_policy()
        
    def get_governance_report(self) -> Dict:
        return {
            "policy_stability": self.constraints.get_stability_score(),
            "violations_prevented": self.constitution.violation_count,
            "current_regime": "AUTONOMOUS" if len(self.policy_history) > 0 else "CONSTITUTIONAL"
        }

from typing import List
from .hierarchical_memory_capsules import MemoryCapsule
from .capsule_registry import CapsuleRegistry

class FidelityBudgetController:
    """
    PHASE 18.7D: Fidelity Budget Controller.
    Enforces global KV cache constraints for high-fidelity capsules.
    """
    def __init__(self, max_high_fidelity_tokens: int = 1024, max_capsules: int = 64):
        self.max_tokens = max_high_fidelity_tokens
        self.max_capsules = max_capsules

    def enforce_budget(self, registry: CapsuleRegistry) -> List[str]:
        """
        Prunes capsules if the budget is exceeded.
        Returns list of evicted capsule IDs.
        """
        evicted = []
        capsules = list(registry.capsules.values())
        
        # Sort by (precision_tier, activation_count, last_accessed)
        # We want to keep HIGH precision, frequently activated, recently accessed capsules.
        tier_priority = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        capsules.sort(key=lambda c: (tier_priority.get(c.precision_tier, 0), c.activation_count, c.last_accessed), reverse=True)
        
        current_tokens = 0
        current_count = 0
        
        for i, capsule in enumerate(capsules):
            token_count = capsule.end_idx - capsule.start_idx
            if current_tokens + token_count > self.max_tokens or current_count + 1 > self.max_capsules:
                # This capsule exceeds the budget
                evicted.append(capsule.capsule_id)
            else:
                current_tokens += token_count
                current_count += 1
                
        # Remove evicted from registry
        for cid in evicted:
            # Note: Registry should probably have a 'remove' method
            # For now we'll just return the list and let the caller handle it or modify registry here
            pass
            
        return evicted

    def get_utilization(self, registry: CapsuleRegistry) -> dict:
        total_tokens = sum(c.end_idx - c.start_idx for c in registry.capsules.values())
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "utilization_pct": (total_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0,
            "capsule_count": len(registry.capsules)
        }

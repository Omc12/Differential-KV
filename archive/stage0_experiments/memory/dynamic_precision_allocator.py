from typing import Dict, List
from .hierarchical_memory_capsules import MemoryCapsule

class DynamicPrecisionAllocator:
    """
    PHASE 18.7D: Dynamic Precision Allocator.
    Assigns fidelity budgets to memory regions based on their classification.
    """
    PRECISION_MAP = {
        "HIGH": 1.0,   # Preserve 100% of tokens in this window
        "MEDIUM": 0.5, # Preserve 50% of tokens (semantic sampling)
        "LOW": 0.1     # Preserve 10% of tokens (aggressive sparse)
    }

    def __init__(self, global_budget_ratio: float = 0.2):
        self.global_budget_ratio = global_budget_ratio

    def allocate_resolution(self, capsule: MemoryCapsule) -> float:
        """Determines the sampling rate or bit-depth for a capsule."""
        return self.PRECISION_MAP.get(capsule.precision_tier, 0.1)

    def classify_region(self, entropy_score: float, tags: List[str]) -> str:
        """Heuristically determines the precision tier for a region."""
        if any(tag in ["ID", "API", "CODE", "EQUATION"] for tag in tags):
            return "HIGH"
        if entropy_score > 3.0:
            return "HIGH"
        if entropy_score > 1.5:
            return "MEDIUM"
        return "LOW"

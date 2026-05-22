
import torch
from typing import Dict, Any, List, Optional

class AdaptiveResidencyBudgetAllocator:
    """
    PHASE 23.4: CRS - Adaptive Residency Budget Allocator.
    Balances residency memory budget and manages compression quotas.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_budget = config.get("base_residency_budget", 1.0) # Normalized
        
        self.metrics = {
            "residency_budget_health": 1.0,
            "allocated_vram_share": 0.0,
            "budget_utilization": 0.0
        }

    def allocate_budget(self, 
                        system_pressure: float, 
                        cognitive_demand: float) -> float:
        """
        Calculates the available residency budget based on pressure and demand.
        """
        # Under pressure, shrink budget. Under demand, expand budget (within limits).
        available_budget = self.base_budget * (1.0 - system_pressure * 0.5)
        available_budget *= (1.0 + cognitive_demand * 0.3)
        
        # Clamp to [0.2, 1.5]
        final_budget = max(0.2, min(1.5, available_budget))
        
        self.metrics["residency_budget_health"] = 1.0 - (system_pressure * 0.2)
        self.metrics["budget_utilization"] = cognitive_demand / (final_budget + 1e-9)
        self.metrics["allocated_vram_share"] = final_budget * 0.1 # Mock share
        
        return final_budget

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

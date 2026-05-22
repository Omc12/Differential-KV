"""
runtime/dynamic_geometry_budget.py
Phase 27: Adaptive Cognitive Routing (ACR)
Manages geometric resource allocation (anchors, rank) across regimes.
"""

from typing import Dict, Any

class DynamicGeometryBudget:
    def __init__(self, total_overhead_limit: float = 0.06):
        self.total_overhead_limit = total_overhead_limit # 6% target
        
    def allocate_budget(self, regime_info: Dict[str, Any], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocates geometry budget based on regime and predicted intent.
        """
        regime = regime_info.get("best_regime", "mixed_mode")
        preservation_req = intent_info.get("geometry_preservation_req", "medium")
        
        # Base anchor density (percentage of tokens that are anchors)
        # We want total overhead < 6%
        # Overhead approx = anchor_density * (1 + extra_metadata)
        
        if regime == "mathematical_reasoning" or preservation_req == "high":
            anchor_density = 0.04 # 4%
            target_rank_ratio = 0.8
            preservation_priority = "high_curvature_pivots"
        elif regime == "recursive_planning":
            anchor_density = 0.035 # 3.5%
            target_rank_ratio = 0.7
            preservation_priority = "attractor_hubs"
        elif regime == "code_generation":
            anchor_density = 0.03 # 3%
            target_rank_ratio = 0.6
            preservation_priority = "syntax_pivots"
        elif regime == "retrieval_heavy":
            anchor_density = 0.02 # 2%
            target_rank_ratio = 0.4
            preservation_priority = "semantic_keys"
        elif regime == "narrative_dialogue":
            anchor_density = 0.01 # 1%
            target_rank_ratio = 0.2
            preservation_priority = "low_risk_regions"
        else:
            anchor_density = 0.025
            target_rank_ratio = 0.5
            preservation_priority = "balanced"
            
        # Ensure we don't exceed total overhead
        # If density > limit, scale down
        if anchor_density > self.total_overhead_limit * 0.8:
            anchor_density = self.total_overhead_limit * 0.8
            
        return {
            "anchor_density": anchor_density,
            "target_rank_ratio": target_rank_ratio,
            "preservation_priority": preservation_priority,
            "estimated_overhead": anchor_density * 1.2, # accounting for metadata
            "budget_status": "optimal" if anchor_density < self.total_overhead_limit else "constrained"
        }

if __name__ == "__main__":
    budgeter = DynamicGeometryBudget()
    math_budget = budgeter.allocate_budget({"best_regime": "mathematical_reasoning"}, {"geometry_preservation_req": "high"})
    print(f"Math Budget: {math_budget}")
    
    dialogue_budget = budgeter.allocate_budget({"best_regime": "narrative_dialogue"}, {"geometry_preservation_req": "low"})
    print(f"Dialogue Budget: {dialogue_budget}")

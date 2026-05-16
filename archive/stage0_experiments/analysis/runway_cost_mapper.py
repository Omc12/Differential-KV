import torch

class RunwayCostMapper:
    """
    PHASE 19.0E: Runway Cost Mapper.
    Maps the performance cost of virtual runways relative to the 
    continuity gains they provide.
    """
    def __init__(self):
        self.history = []

    def log_cost_benefit(self, runway_size: int, continuity_gain: float, tps_impact: float):
        self.history.append({
            "runway_size": runway_size,
            "continuity_gain": continuity_gain,
            "tps_impact": tps_impact,
            "efficiency": continuity_gain / (tps_impact + 1e-6)
        })

    def get_optimal_budget(self) -> int:
        if not self.history:
            return 32 # Default
            
        # Find budget with highest efficiency
        best = max(self.history, key=lambda x: x["efficiency"])
        return best["runway_size"]

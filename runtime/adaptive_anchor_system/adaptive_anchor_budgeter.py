import torch

class AdaptiveAnchorBudgeter:
    """
    Manages the global anchor budget based on current TPS (Tokens Per Second)
    and hardware latency constraints.
    """
    def __init__(self, target_overhead: float = 0.05, max_total_anchors: int = 2048):
        self.target_overhead = target_overhead
        self.max_total_anchors = max_total_anchors
        self.current_budget = max_total_anchors

    def adjust_budget(self, current_tps: float, baseline_tps: float):
        """
        Reduces budget if adaptive anchoring is causing too much degradation.
        """
        if baseline_tps <= 0:
            return self.current_budget
            
        overhead = 1.0 - (current_tps / baseline_tps)
        
        if overhead > self.target_overhead * 1.5:
            # Over budget, reduce anchor density
            self.current_budget = int(self.current_budget * 0.9)
        elif overhead < self.target_overhead * 0.8:
            # Under budget, can afford more anchors for stability
            self.current_budget = min(self.max_total_anchors, int(self.current_budget * 1.1))
            
        return self.current_budget

    def enforce_budget(self, anchor_indices: torch.Tensor, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Prunes lower-importance anchors to fit within the current budget.
        """
        if anchor_indices.size(0) <= self.current_budget:
            return anchor_indices
            
        # Select top-K based on importance
        _, top_idx = torch.topk(importance_scores, self.current_budget)
        return anchor_indices[top_idx]

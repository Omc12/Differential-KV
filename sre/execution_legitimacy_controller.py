
import torch
from typing import Dict, Any

class ExecutionLegitimacyController:
    """
    PHASE 22.0: SRE - Execution Legitimacy Controller.
    Governs sparse stability and prevents catastrophic sparse collapse.
    """
    def __init__(self, min_active_ratio: float = 0.1):
        self.min_active_ratio = min_active_ratio
        self.stability_window = []
        self.legitimacy_metrics = {
            "over_pruning_events": 0,
            "stability_violations": 0,
            "mean_legitimacy_score": 1.0
        }

    def validate_execution_plan(self, 
                                 active_mask: torch.Tensor, 
                                 symbolic_importance: float) -> torch.Tensor:
        """
        Intervenes if the proposed execution plan is too aggressive (over-pruning).
        """
        active_ratio = active_mask.float().mean().item()
        
        # If we are pruning too much relative to symbolic importance
        required_ratio = self.min_active_ratio * (1.0 + symbolic_importance)
        
        if active_ratio < required_ratio:
            self.legitimacy_metrics["over_pruning_events"] += 1
            # Emergency activation: force-enable some regions
            # Here we just boost the mask
            boosted_mask = active_mask.float() + (torch.rand_like(active_mask.float()) * 0.2)
            return boosted_mask > 0.5
            
        return active_mask

    def check_stability(self, entropy_health: float) -> bool:
        """
        Checks if the sparse runtime is oscillating or collapsing.
        """
        self.stability_window.append(entropy_health)
        if len(self.stability_window) > 10:
            self.stability_window.pop(0)
            
        if len(self.stability_window) < 5:
            return True
            
        # Check for sudden collapse in entropy
        avg_entropy = sum(self.stability_window[:-1]) / (len(self.stability_window) - 1)
        current_entropy = self.stability_window[-1]
        
        if current_entropy < 0.3 * avg_entropy:
            self.legitimacy_metrics["stability_violations"] += 1
            return False
            
        return True

    def get_legitimacy_score(self) -> float:
        """
        Calculates overall execution legitimacy.
        1.0 is perfect, < 0.5 is risky.
        """
        penalty = (self.legitimacy_metrics["over_pruning_events"] * 0.05 + 
                   self.legitimacy_metrics["stability_violations"] * 0.2)
        score = max(0.0, 1.0 - penalty)
        self.legitimacy_metrics["mean_legitimacy_score"] = score
        return score

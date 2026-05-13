import torch
from typing import Dict

class AdaptiveAnchorBudgeter:
    """
    PHASE 7.5A: Adaptive Anchor Budgeter
    Prevents over-allocation of anchors by balancing retrieval survival 
    against available VRAM and TPS targets.
    """
    def __init__(self, max_vram_gb: float = 2.0, tps_target: float = 100.0):
        self.max_vram_gb = max_vram_gb
        self.tps_target = tps_target
        self.anchor_size_bytes = 1024 # Estimated size per anchor head

    def calculate_budget(self, current_vram_usage_gb: float, current_tps: float) -> int:
        """
        Determines the maximum allowed number of anchors.
        """
        vram_headroom = self.max_vram_gb - current_vram_usage_gb
        
        # Calculate budget based on VRAM constraints
        max_anchors_vram = int((vram_headroom * 1024**3) / self.anchor_size_bytes)
        
        # Calculate budget based on TPS impact (empirical heuristic)
        # If TPS drops below target, we reduce anchor density
        tps_scaling = min(1.0, current_tps / self.tps_target) if current_tps > 0 else 1.0
        
        # Combined budget
        final_budget = int(max_anchors_vram * tps_scaling)
        return max(512, final_budget) # Hard floor for stability

    def should_prune(self, current_count: int, budget: int) -> bool:
        """Checks if current anchor count exceeds the allowed budget."""
        return current_count > budget

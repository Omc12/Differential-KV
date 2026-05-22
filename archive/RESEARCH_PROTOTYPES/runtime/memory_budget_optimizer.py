"""
runtime/memory_budget_optimizer.py

Dynamically allocates VRAM/resources for Differential KV.
Supports rank scaling, adaptive anchor density, and dynamic repair budgets.
"""

import numpy as np
from typing import Dict, List, Any, Optional

class MemoryBudgetOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.total_vram_limit = config.get("vram_limit_gb", 8) * 1024 * 1024 * 1024 # bytes
        self.base_rank = config.get("base_rank", 16)
        self.max_rank = config.get("max_rank", 64)
        self.min_rank = config.get("min_rank", 4)
        
        self.current_vram_usage = 0
        self.budget_allocation = {
            "anchors": 0.4,    # 40% for SAM anchors
            "lowrank": 0.4,    # 40% for low-rank deltas
            "repair": 0.1,     # 10% for active repairs (ACTR)
            "overhead": 0.1    # 10% overhead
        }

    def allocate_resources(self, cognitive_state: Dict[str, Any], context_depth: int) -> Dict[str, Any]:
        """
        Determines the optimal resource allocation based on current cognitive state.
        Returns: {
            "target_rank": int,
            "anchor_density": float,
            "repair_budget": float,
            "high_fidelity_window": bool
        }
        """
        collapse_prob = cognitive_state.get("collapse_probability", 0.0)
        stability = cognitive_state.get("manifold_stability", 1.0)
        drift = cognitive_state.get("latent_drift", 0.0)
        
        # 1. Rank Scaling
        # Increase rank if stability is low or collapse risk is high
        rank_factor = 1.0 + (collapse_prob * 2.0) + (max(0, 0.5 - stability) * 2.0)
        target_rank = int(np.clip(self.base_rank * rank_factor, self.min_rank, self.max_rank))
        
        # 2. Anchor Density
        # More anchors if reasoning is complex (high drift/low stability)
        anchor_factor = 1.0 + (drift * 3.0) + (1.0 - stability)
        anchor_density = np.clip(0.1 * anchor_factor, 0.05, 0.5) # 5% to 50%
        
        # 3. Repair Budget
        # Reserve more for repairs if we are near a "cognitive cliff"
        repair_budget = np.clip(0.05 + collapse_prob * 0.2, 0.05, 0.25)
        
        # 4. High Fidelity Window
        # Enable high fidelity if we are in a critical reasoning phase (high curvature)
        curvature = cognitive_state.get("curvature", 0.0)
        high_fidelity_window = curvature > 0.4 or collapse_prob > 0.6
        
        return {
            "target_rank": target_rank,
            "anchor_density": anchor_density,
            "repair_budget": repair_budget,
            "high_fidelity_window": high_fidelity_window
        }

    def update_vram_usage(self, usage_bytes: int):
        self.current_vram_usage = usage_bytes
        
    def check_pressure(self) -> float:
        """Returns memory pressure from 0.0 to 1.0."""
        return np.clip(self.current_vram_usage / self.total_vram_limit, 0, 1)

import torch
import numpy as np
from typing import Dict, List, Optional

class AdaptiveAnchorOptimizer:
    """
    PHASE 7.5A: Adaptive Anchor Optimizer
    Dynamically adjusts anchor density and positioning based on real-time 
    retrieval success and collision metrics.
    """
    def __init__(
        self, 
        base_spacing: int = 128,
        min_spacing: int = 32,
        max_spacing: int = 1024,
        target_survival_rate: float = 0.99
    ):
        self.base_spacing = base_spacing
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.target_survival_rate = target_survival_rate
        
        self.current_spacing = base_spacing
        self.retrieval_history = []
        self.collision_history = []

    def optimize_layout(
        self, 
        retrieval_metrics: Dict[str, float],
        collision_metrics: Dict[str, float]
    ) -> int:
        """
        Calculates new anchor spacing based on recent performance.
        Returns the optimized spacing value.
        """
        survival_rate = retrieval_metrics.get("survival_rate", 1.0)
        collision_rate = collision_metrics.get("collision_rate", 0.0)
        
        # Gradual adjustment to prevent oscillation
        adjustment_factor = 1.0
        
        if survival_rate < self.target_survival_rate:
            # Increase density (reduce spacing) if survival is low
            adjustment_factor = 0.8
        elif collision_rate > 0.05:
            # Slightly decrease density if collisions are high
            # but only if survival is stable
            if survival_rate >= self.target_survival_rate:
                adjustment_factor = 1.1
        
        new_spacing = int(self.current_spacing * adjustment_factor)
        self.current_spacing = max(self.min_spacing, min(self.max_spacing, new_spacing))
        
        return self.current_spacing

    def get_anchor_indices(self, sequence_length: int) -> torch.Tensor:
        """
        Generates anchor indices based on the current optimized spacing.
        """
        indices = torch.arange(0, sequence_length, self.current_spacing)
        return indices

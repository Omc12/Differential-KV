"""
runtime/sparse_geometry_scheduler.py
Phase 23: Sparse Geometry Scheduler (SGS)
Enables geometry preservation ONLY near reasoning pivots and instability.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any

class SparseGeometryScheduler:
    """
    Manages the budget for geometric preservation.
    """
    def __init__(self, 
                 max_geometry_overhead: float = 0.05, 
                 instability_threshold: float = 0.6):
        self.max_overhead = max_geometry_overhead
        self.instability_threshold = instability_threshold
        self.current_usage = 0.0

    def should_save_geometry(self, 
                             is_pivot: bool, 
                             instability_score: float, 
                             curvature: float) -> bool:
        """
        Decision logic for when to store rich geometric data.
        """
        # Always save at pivots if budget allows
        if is_pivot and self.current_usage < self.max_overhead:
            return True
            
        # Save if instability is high
        if instability_score > self.instability_threshold:
            return True
            
        # Save during high-curvature spans
        if curvature > 3.0:
            return True
            
        return False

    def update_budget(self, num_geometry_anchors: int, total_tokens: int):
        """Tracks geometric overhead."""
        # Estimate: each geometry anchor is ~4x size of standard anchor
        self.current_usage = (num_geometry_anchors * 4.0) / (total_tokens + 1e-6)

class RecursiveReasoningDetector:
    """
    Detects patterns indicating recursive or complex logical reasoning.
    """
    def is_recursive(self, tokens: List[int], attention_patterns: torch.Tensor) -> bool:
        # High self-attention loops often indicate recursion
        return False

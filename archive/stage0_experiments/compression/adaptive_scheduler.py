"""
compression/adaptive_scheduler.py
Phase 10: Intelligence-Aware Adaptive Scheduling for Differential KV.
Supports Layer-wise, Head-wise, and Temporal adaptive policies.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

class AdaptiveScheduler:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        base_rank: int = 8,
        max_rank: int = 64,
        sensitivity_map: Optional[Dict] = None,
        temporal_alpha: float = 0.1,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.base_rank = base_rank
        self.max_rank = max_rank
        self.sensitivity_map = sensitivity_map or {}
        self.temporal_alpha = temporal_alpha
        
        # State tracking for temporal stability
        self.drift_history = []
        self.current_rank_boost = 0.0
        
        # Pre-process sensitivity maps for fast lookup
        self.layer_sens = np.ones(num_layers)
        if "layer_sensitivity" in self.sensitivity_map:
            for item in self.sensitivity_map["layer_sensitivity"]:
                l = item["layer"]
                # Normalize KL divergence to a multiplier (e.g., 0.5 to 2.0)
                kl = item["kl_divergence"]
                self.layer_sens[l] = max(0.5, min(2.0, kl * 5.0)) # Heuristic scaling

        self.head_sens = np.ones((num_layers, num_heads))
        if "head_sensitivity" in self.sensitivity_map:
            for item in self.sensitivity_map["head_sensitivity"]:
                l, h = item["layer"], item["head"]
                kl = item["kl_divergence"]
                self.head_sens[l, h] = max(0.5, min(2.0, kl * 5.0))

    def get_layer_rank(self, layer_idx: int, step_idx: int = 0) -> int:
        """
        Calculates rank for a specific layer, considering its sensitivity and current temporal state.
        """
        sens = self.layer_sens[layer_idx]
        temporal_boost = self.current_rank_boost * sens
        
        rank = int(self.base_rank * sens + temporal_boost)
        return int(min(max(rank, 4), self.max_rank))

    def get_head_rank(self, layer_idx: int, head_idx: int, step_idx: int = 0) -> int:
        """
        Calculates rank for a specific head.
        """
        sens = self.head_sens[layer_idx, head_idx]
        layer_rank = self.get_layer_rank(layer_idx, step_idx)
        
        rank = int(layer_rank * sens)
        return int(min(max(rank, 2), self.max_rank))

    def update_temporal_state(self, drift_signal: float):
        """
        Updates the global rank boost based on detected semantic drift.
        drift_signal: KL divergence, entropy spike, etc.
        """
        self.drift_history.append(drift_signal)
        
        if len(self.drift_history) > 1:
            delta = self.drift_history[-1] - self.drift_history[-2]
            if delta > 0:
                self.current_rank_boost += self.temporal_alpha * delta * 10.0 # Aggressive response
            else:
                self.current_rank_boost *= 0.98 # Gradual decay
                
        self.current_rank_boost = max(0.0, min(self.current_rank_boost, float(self.max_rank - self.base_rank)))

    def get_allocation_map(self, step_idx: int = 0) -> Dict[int, List[int]]:
        """
        Generates a full allocation map for all layers and heads.
        """
        alloc_map = {}
        for l in range(self.num_layers):
            head_ranks = [self.get_head_rank(l, h, step_idx) for h in range(self.num_heads)]
            alloc_map[l] = head_ranks
        return alloc_map

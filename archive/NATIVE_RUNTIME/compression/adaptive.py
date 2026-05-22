"""
compression/adaptive.py

Phase 7: Adaptive Rank Selection logic for Shared-Basis Differential KV.
Determines the optimal rank for a given delta block.
"""

import torch
from typing import List, Optional, Union

class AdaptiveRankSelector:
    def __init__(
        self, 
        rank_buckets: List[int] = [8, 16, 32, 64, 128],
        method: str = "energy",
        target_energy: float = 0.95
    ):
        self.rank_buckets = sorted(rank_buckets)
        self.method = method
        self.target_energy = target_energy

    def select_rank(self, deltas: torch.Tensor) -> int:
        """
        Selects a rank from rank_buckets based on the chosen heuristic.
        """
        if self.method == "fixed":
            return self.rank_buckets[-1] # Use largest if fixed requested but called through here
        
        if self.method == "energy":
            return self._select_by_energy(deltas)
        elif self.method == "variance":
            return self._select_by_variance(deltas)
        elif self.method == "norm":
            return self._select_by_norm(deltas)
        else:
            return self.rank_buckets[len(self.rank_buckets)//2] # Default to middle

    def _select_by_energy(self, deltas: torch.Tensor) -> int:
        """
        Expensive but accurate: uses SVD singular values to determine rank.
        """
        # deltas: [n, d]
        try:
            # We only need singular values
            S = torch.linalg.svdvals(deltas.float())
            energy = torch.cumsum(S**2, dim=0)
            total_energy = energy[-1]
            
            target = total_energy * self.target_energy
            
            # Find first index where energy >= target
            idx = torch.where(energy >= target)[0]
            if len(idx) > 0:
                required_rank = idx[0].item() + 1
            else:
                required_rank = len(S)
                
            # Snap to buckets
            for rb in self.rank_buckets:
                if rb >= required_rank:
                    return rb
            return self.rank_buckets[-1]
        except Exception:
            return self.rank_buckets[len(self.rank_buckets)//2]

    def _select_by_variance(self, deltas: torch.Tensor) -> int:
        """
        Fast proxy: higher variance often implies more complex structure.
        Uses empirical thresholds (calibrated for KV deltas).
        """
        var = torch.var(deltas).item()
        # Heuristic thresholds (placeholders, should be tuned)
        if var < 0.001: return self.rank_buckets[0]
        if var < 0.01:  return self.rank_buckets[1]
        if var < 0.05:  return self.rank_buckets[2]
        return self.rank_buckets[-1]

    def _select_by_norm(self, deltas: torch.Tensor) -> int:
        """
        Fast proxy: maximum absolute value.
        """
        m = torch.max(torch.abs(deltas)).item()
        if m < 0.05: return self.rank_buckets[0]
        if m < 0.2:  return self.rank_buckets[1]
        if m < 0.5:  return self.rank_buckets[2]
        return self.rank_buckets[-1]

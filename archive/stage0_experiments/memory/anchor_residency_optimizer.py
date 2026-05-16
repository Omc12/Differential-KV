"""
memory/anchor_residency_optimizer.py

Optimizes which anchors stay resident in high-pressure scenarios.
Uses a hybrid frequency-recency-importance (FRI) model.
"""

from typing import Dict, List, Any
import logging

class AnchorResidencyOptimizer:
    """
    Intelligent residency manager for extreme contexts.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stats: Dict[int, Dict[str, Any]] = {} # shard_id -> stats
        self.logger = logging.getLogger("AnchorResidencyOptimizer")

    def update_stats(self, shard_id: int, frequency: int, importance: float):
        """Updates stats for a shard."""
        if shard_id not in self.stats:
            self.stats[shard_id] = {"freq": 0, "imp": 0.0, "last_access": 0}
            
        self.stats[shard_id]["freq"] += frequency
        self.stats[shard_id]["imp"] = importance
        # self.stats[shard_id]["last_access"] = time.time()

    def get_residency_priority(self, shard_id: int) -> float:
        """
        Calculates priority score based on Frequency * Importance.
        """
        if shard_id not in self.stats: return 0.0
        s = self.stats[shard_id]
        return s["freq"] * s["imp"]

    def optimize_residency(self) -> List[int]:
        """
        Returns the top 'capacity' shards that should remain resident.
        """
        sorted_shards = sorted(
            self.stats.keys(), 
            key=lambda x: self.get_residency_priority(x), 
            reverse=True
        )
        return sorted_shards[:self.capacity]

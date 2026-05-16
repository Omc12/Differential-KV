"""
distributed/distributed_hotset_router.py

Specialized router for high-frequency "hot" retrieval sets.
Bypasses standard sharding for blocks that are globally relevant.
"""

import torch
from typing import Set, Dict, Any
import logging

class DistributedHotsetRouter:
    """
    Identifies and manages 'hot' shards that should be replicated 
    rather than sharded for maximum throughput.
    """
    def __init__(self, hot_threshold: int = 100):
        self.hot_threshold = hot_threshold
        self.hot_shards: Set[int] = set()
        self.access_history: Dict[int, int] = {}
        self.logger = logging.getLogger("DistributedHotsetRouter")

    def track_access(self, shard_id: int):
        """Tracks access to identify hot shards."""
        self.access_history[shard_id] = self.access_history.get(shard_id, 0) + 1
        
        if self.access_history[shard_id] >= self.hot_threshold:
            if shard_id not in self.hot_shards:
                self.hot_shards.add(shard_id)
                self.logger.info(f"Shard {shard_id} promoted to HOTSET (Access: {self.access_history[shard_id]})")

    def is_hot(self, shard_id: int) -> bool:
        """Checks if a shard is in the hotset."""
        return shard_id in self.hot_shards

    def get_hotset_routing_plan(self) -> Dict[int, str]:
        """
        Returns a plan indicating that hot shards should be BROADCAST 
        or replicated across all nodes.
        """
        return {shard_id: "REPLICATE" for shard_id in self.hot_shards}

    def cooling_phase(self):
        """Periodically cools access counts to allow hotset to change."""
        for shard_id in list(self.access_history.keys()):
            self.access_history[shard_id] //= 2
            if self.access_history[shard_id] < self.hot_threshold // 2:
                if shard_id in self.hot_shards:
                    self.hot_shards.remove(shard_id)
                    self.logger.info(f"Shard {shard_id} cooled and removed from HOTSET")

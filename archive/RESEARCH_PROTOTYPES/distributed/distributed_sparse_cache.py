"""
distributed/distributed_sparse_cache.py

A distributed cache for hot sparse KV blocks.
Reduces redundant transfers by caching frequently accessed blocks.
"""

import torch
from typing import Dict, List, Optional
import time
import logging

class DistributedSparseCache:
    """
    LRU Cache for sparse KV blocks across the distributed runtime.
    """
    def __init__(self, capacity_gb: float, device: torch.device):
        self.capacity_bytes = capacity_gb * 1024 * 1024 * 1024
        self.device = device
        self.cache: Dict[int, torch.Tensor] = {} # shard_id -> block
        self.access_times: Dict[int, float] = {}
        self.current_size_bytes = 0
        self.logger = logging.getLogger("DistributedSparseCache")

    def get(self, shard_id: int) -> Optional[torch.Tensor]:
        """Retrieves a block from cache, updating access time."""
        if shard_id in self.cache:
            self.access_times[shard_id] = time.time()
            return self.cache[shard_id]
        return None

    def put(self, shard_id: int, block: torch.Tensor):
        """Puts a block into cache, evicting if necessary."""
        block_size = block.element_size() * block.nelement()
        
        while self.current_size_bytes + block_size > self.capacity_bytes and self.cache:
            self.evict_lru()
            
        self.cache[shard_id] = block.to(self.device)
        self.access_times[shard_id] = time.time()
        self.current_size_bytes += block_size
        self.logger.debug(f"Cached Shard {shard_id}, Size: {block_size/1024/1024:.2f} MB")

    def evict_lru(self):
        """Evicts the least recently used block."""
        if not self.access_times:
            return
            
        lru_shard = min(self.access_times, key=self.access_times.get)
        block = self.cache.pop(lru_shard)
        self.current_size_bytes -= block.element_size() * block.nelement()
        del self.access_times[lru_shard]
        self.logger.debug(f"Evicted Shard {lru_shard} from cache")

    def get_cache_metrics(self) -> Dict[str, float]:
        """Returns cache hit rate and occupancy."""
        return {
            "occupancy_pct": (self.current_size_bytes / self.capacity_bytes) * 100,
            "shard_count": len(self.cache),
            "total_size_mb": self.current_size_bytes / 1024 / 1024
        }

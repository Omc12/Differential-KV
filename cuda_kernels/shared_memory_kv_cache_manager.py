import torch
from typing import Dict, List, Any
import logging

class SharedMemoryKVCacheManager:
    """
    Manages fast staging of KV segments in GPU Shared Memory (SRAM).
    """
    def __init__(self, capacity_mb: int = 48): # Typical A100 shared memory per SM
        self.capacity_mb = capacity_mb
        self.staged_segments: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger("SharedMemoryKVCacheManager")

    def stage_segment(self, segment_id: str, size_mb: float):
        """Stages a segment in simulated shared memory."""
        current_total = sum(self.staged_segments.values())
        if current_total + size_mb > self.capacity_mb:
            # LRU eviction
            oldest = min(self.staged_segments, key=self.staged_segments.get)
            del self.staged_segments[oldest]
            self.logger.info(f"Evicted {oldest} from shared memory.")
            
        self.staged_segments[segment_id] = size_mb
        self.logger.info(f"Staged {segment_id} in shared memory.")

    def access_segment(self, segment_id: str) -> bool:
        """Checks if a segment is in shared memory (hit) or needs HBM access (miss)."""
        if segment_id in self.staged_segments:
            self.hits += 1
            return True
        self.misses += 1
        return False

    def get_cache_metrics(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "shared_memory_hit_rate": self.hits / max(1, total),
            "shared_memory_usage_mb": sum(self.staged_segments.values())
        }

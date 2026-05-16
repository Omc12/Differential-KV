"""
runtime/latency_aware_anchor_allocator.py

Latency-aware allocator for anchors in Differential KV.
Optimizes anchor placement to minimize retrieval latency and synchronization overhead.
"""

import torch
from typing import List, Dict

class LatencyAwareAnchorAllocator:
    def __init__(self, vram_limit_mb: int = 2048):
        self.vram_limit = vram_limit_mb
        self.allocated_mb = 0
        self.anchor_map = {} # id -> (vram_offset, importance)
        self.latency_penalty = {} # id -> historical latency cost

    def allocate_anchor_block(self, anchor_id: str, size_mb: float, importance: float):
        """
        Allocates a block of VRAM for an anchor, considering its importance and latency cost.
        """
        if self.allocated_mb + size_mb > self.vram_limit:
            # Evict least important / highest latency anchor
            self._evict_under_pressure()
            
        self.anchor_map[anchor_id] = {"size": size_mb, "importance": importance}
        self.allocated_mb += size_mb
        return True

    def update_latency_feedback(self, anchor_id: str, latency_ms: float):
        """Updates the latency penalty for an anchor to inform future allocation decisions."""
        self.latency_penalty[anchor_id] = latency_ms

    def _evict_under_pressure(self):
        """Simple eviction heuristic: Importance / Latency score."""
        if not self.anchor_map:
            return
            
        def score_fn(aid):
            importance = self.anchor_map[aid]["importance"]
            latency = self.latency_penalty.get(aid, 1.0)
            return importance / (latency + 1e-6)
            
        victim_id = min(self.anchor_map.keys(), key=score_fn)
        self.allocated_mb -= self.anchor_map[victim_id]["size"]
        del self.anchor_map[victim_id]
        if victim_id in self.latency_penalty:
            del self.latency_penalty[victim_id]

    def get_allocation_stats(self):
        """Returns VRAM utilization and eviction history."""
        return {
            "vram_utilization": self.allocated_mb / self.vram_limit,
            "anchor_count": len(self.anchor_map),
            "total_allocated_mb": self.allocated_mb
        }

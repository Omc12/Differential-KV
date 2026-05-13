"""
memory/hierarchical_sparse_cache.py

Hierarchical cache manager for sparse KV components.
Manages tiers: Fast L1 (Registers/SRAM), L2 (L2 Cache), and Global VRAM.
"""

import torch

class HierarchicalSparseCache:
    def __init__(self, l1_size: int = 1024, l2_size: int = 4096):
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l1_cache = {} # Logical ID -> Physical Offset
        self.l2_cache = {}
        self.vram_pool = {}

    def access_kv(self, logical_id: int):
        """Attempts to access KV from the fastest tier possible."""
        if logical_id in self.l1_cache:
            return "L1_HIT"
        elif logical_id in self.l2_cache:
            self._promote_to_l1(logical_id)
            return "L2_HIT"
        elif logical_id in self.vram_pool:
            self._promote_to_l2(logical_id)
            return "VRAM_HIT"
        return "MISS"

    def _promote_to_l1(self, logical_id: int):
        """Promotes a KV block from L2 to L1 cache."""
        if len(self.l1_cache) >= self.l1_size:
            # Simple LRU eviction (mocked)
            evicted = next(iter(self.l1_cache))
            del self.l1_cache[evicted]
            self.l2_cache[evicted] = True
            
        self.l1_cache[logical_id] = True
        if logical_id in self.l2_cache:
            del self.l2_cache[logical_id]

    def _promote_to_l2(self, logical_id: int):
        """Promotes a KV block from Global VRAM to L2 cache."""
        if len(self.l2_cache) >= self.l2_size:
            evicted = next(iter(self.l2_cache))
            del self.l2_cache[evicted]
            self.vram_pool[evicted] = True
            
        self.l2_cache[logical_id] = True

    def get_cache_stats(self):
        """Returns hit rates and occupancy for each cache tier."""
        return {
            "l1_occupancy": len(self.l1_cache) / self.l1_size,
            "l2_occupancy": len(self.l2_cache) / self.l2_size,
            "vram_pool_size": len(self.vram_pool)
        }

"""
memory/shared_resonance_cache.py

Manages shared resonance states and persistent geometry buffers for 
long-horizon cognitive stability.
"""

import torch
from typing import Dict, Any

class SharedResonanceCache:
    """
    A global cache for resonance states, allowing different agents or 
    heads to share manifold information.
    """
    def __init__(self, resonance_rank: int, max_entries: int = 1000):
        self.resonance_rank = resonance_rank
        self.max_entries = max_entries
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_counts: Dict[str, int] = {}

    def update(self, key: str, resonance_vector: torch.Tensor):
        """
        Updates the cache with a new resonance vector.
        """
        self.cache[key] = resonance_vector.detach()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Eviction logic (LRU-lite)
        if len(self.cache) > self.max_entries:
            oldest = min(self.access_counts, key=self.access_counts.get)
            del self.cache[oldest]
            del self.access_counts[oldest]

    def fetch(self, key: str) -> torch.Tensor:
        """
        Fetches a resonance vector from the cache.
        Returns a zero vector if not found.
        """
        if key in self.cache:
            self.access_counts[key] += 1
            return self.cache[key]
        return None

class PersistentGeometryBuffer:
    """
    Maintains stable geometry manifolds across long context windows.
    Stores compressed attractors that represent the "core" of the conversation.
    """
    def __init__(self, capacity: int, feat_dim: int):
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.buffer = torch.zeros((capacity, feat_dim), device="cuda" if torch.cuda.is_available() else "cpu")
        self.pointer = 0

    def store_attractor(self, attractor: torch.Tensor):
        """
        Stores a new geometric attractor in the persistent buffer.
        """
        num_vecs = attractor.shape[0]
        if self.pointer + num_vecs > self.capacity:
            # Wrap around or resize
            self.pointer = 0 
            
        self.buffer[self.pointer:self.pointer+num_vecs] = attractor
        self.pointer += num_vecs

    def get_manifold_hull(self):
        """
        Returns the convex hull (or simplified representation) of the stored geometry.
        """
        return self.buffer[:self.pointer]

if __name__ == "__main__":
    cache = SharedResonanceCache(16)
    cache.update("regime_math", torch.randn(16))
    print(f"Cache Fetch: {cache.fetch('regime_math') is not None}")
    
    buffer = PersistentGeometryBuffer(100, 64)
    buffer.store_attractor(torch.randn(5, 64))
    print(f"Buffer Pointer: {buffer.pointer}")

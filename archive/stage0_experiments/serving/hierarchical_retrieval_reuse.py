import torch
from typing import Dict, Optional

class HierarchicalRetrievalReuse:
    """
    Caches and reuses recently retrieved sparse KV blocks to improve locality.
    """
    def __init__(self, l1_size: int = 1024, l2_size: int = 4096):
        self.l1_cache: Dict[int, torch.Tensor] = {} # Most recent
        self.l2_cache: Dict[int, torch.Tensor] = {} # Recent
        self.hits = 0
        self.misses = 0

    def get_blocks(self, block_indices: torch.Tensor) -> Optional[torch.Tensor]:
        # Check if blocks are in L1 or L2
        # Mocking the hit/miss logic for the purpose of the runtime integration
        batch_size = block_indices.shape[0]
        # Simulate hit rate
        hit_prob = 0.8
        if torch.rand(1).item() < hit_prob:
            self.hits += 1
            return torch.randn((batch_size, 2, 32, 128), device="cuda", dtype=torch.float16)
        else:
            self.misses += 1
            return None

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hits": self.hits,
            "misses": self.misses
        }

from typing import Dict, List, Optional
from .hierarchical_memory_capsules import MemoryCapsule
import torch

class CapsuleRegistry:
    """
    PHASE 18.7A: Global Capsule Registry.
    Provides lookup and indexing services for active memory capsules.
    """
    def __init__(self):
        self.capsules: Dict[str, MemoryCapsule] = {}
        self.index_map: Dict[int, str] = {} # token_idx -> capsule_id

    def register(self, capsule: MemoryCapsule):
        self.capsules[capsule.capsule_id] = capsule
        for idx in range(capsule.start_idx, capsule.end_idx):
            self.index_map[idx] = capsule.capsule_id

    def get_capsule_for_token(self, token_idx: int) -> Optional[MemoryCapsule]:
        cid = self.index_map.get(token_idx)
        return self.capsules.get(cid) if cid else None

    def get_all_indices(self) -> torch.Tensor:
        """Returns all indices currently protected by capsules."""
        if not self.index_map:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(list(self.index_map.keys()), dtype=torch.long)

    def filter_by_tier(self, tier: str) -> List[MemoryCapsule]:
        return [c for c in self.capsules.values() if c.precision_tier == tier]

    def clear(self):
        self.capsules.clear()
        self.index_map.clear()

    def __len__(self):
        return len(self.capsules)

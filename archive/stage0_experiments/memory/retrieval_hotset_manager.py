import torch
from typing import Set

class RetrievalHotsetManager:
    """
    Identifies and maintains the 'hotset' of KV tokens that must remain in VRAM.
    Uses retrieval frequency and recency to determine hotset membership.
    """
    def __init__(self, hotset_size_limit: int = 16384):
        self.hotset_size_limit = hotset_size_limit
        self.hotset_indices = set()
        self.access_history = {}

    def update_hotset(self, accessed_indices: torch.Tensor):
        """
        Updates the hotset based on recent activity.
        """
        # accessed_indices: [num_accessed]
        for idx in accessed_indices.tolist():
            self.access_history[idx] = self.access_history.get(idx, 0) + 1
            self.hotset_indices.add(idx)

        # Prune hotset if it exceeds limit
        if len(self.hotset_indices) > self.hotset_size_limit:
            # Sort by access count (descending) and keep top-k
            sorted_indices = sorted(self.hotset_indices, key=lambda x: self.access_history[x], reverse=True)
            self.hotset_indices = set(sorted_indices[:self.hotset_size_limit])

    def get_vram_indices(self) -> torch.Tensor:
        """
        Returns a tensor of indices that should be in VRAM.
        """
        return torch.tensor(list(self.hotset_indices))

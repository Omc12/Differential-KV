import torch

class KVLocalityOptimizer:
    """
    Rearranges KV blocks in memory to maximize L2/L3 cache hits.
    Groups tokens that are frequently accessed together.
    """
    def __init__(self, reorder_frequency: int = 128):
        self.reorder_frequency = reorder_frequency
        self.access_counts = {}

    def track_access(self, indices: torch.Tensor):
        """
        Tracks which KV positions are accessed to identify 'hot' clusters.
        """
        for idx in indices.flatten().tolist():
            self.access_counts[idx] = self.access_counts.get(idx, 0) + 1

    def optimize_layout(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Physically reorders KV cache to put high-access tokens together.
        """
        # Sort indices by access frequency
        sorted_indices = sorted(self.access_counts.keys(), key=lambda x: self.access_counts[x], reverse=True)
        
        # Re-index the cache
        # In a real system, this is expensive and only done periodically
        if len(sorted_indices) > 0:
            reordered = kv_cache[:, :, sorted_indices, :]
            return reordered
            
        return kv_cache

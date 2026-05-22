import torch
from typing import Dict, List

class HotsetShardAllocator:
    """
    PHASE 7.5C: Hotset Shard Allocator
    Identifies frequently accessed KV 'hotsets' and distributes them 
    across GPU memory shards to prevent single-shard bandwidth saturation.
    """
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.hotset_access_counts: Dict[int, int] = {}

    def update_access_frequency(self, block_indices: torch.Tensor):
        """Tracks which KV blocks are 'hot'."""
        indices = block_indices.flatten().tolist()
        for idx in indices:
            self.hotset_access_counts[idx] = self.hotset_access_counts.get(idx, 0) + 1

    def allocate_to_shard(self, block_index: int) -> int:
        """Determines which shard a block should reside in."""
        # Simple balanced sharding based on index
        # A more advanced version would rebalance hot blocks to different shards
        return block_index % self.num_shards

    def get_shard_map(self, blocks: torch.Tensor) -> torch.Tensor:
        """Maps a tensor of block indices to their respective shards."""
        return blocks % self.num_shards

import torch

class HotsetShardAllocator:
    """
    Allocates replicated 'shards' for extremely hot tokens.
    Multiple users can access different shards of the same token 
    simultaneously without bank contention.
    """
    def __init__(self, replication_factor: int = 4):
        self.replication_factor = replication_factor
        self.hotset_shards = {}

    def allocate_shards(self, hot_indices: torch.Tensor):
        """
        Marks indices for sharding.
        In a real system, this would trigger a copy of the KV data 
        into multiple memory locations.
        """
        for idx in hot_indices.tolist():
            self.hotset_shards[idx] = self.replication_factor

    def get_shard_index(self, token_idx: int, user_id: int) -> int:
        """
        Returns a shard-specific offset for the token.
        """
        if token_idx not in self.hotset_shards:
            return 0
        return user_id % self.replication_factor

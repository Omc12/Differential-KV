import torch
from typing import List, Dict

class RetrievalLocalityPartitioner:
    """
    PHASE 7.5C: Retrieval Locality Partitioner
    Ensures that concurrent retrieval requests from different users 
    are mapped to distinct VRAM regions to minimize L2 cache thrashing 
    and memory bus contention.
    """
    def __init__(self, num_partitions: int = 8):
        self.num_partitions = num_partitions
        self.partition_map: Dict[int, int] = {} # user_id -> partition_id

    def assign_partition(self, user_id: int) -> int:
        """Assigns a persistent partition to a user."""
        if user_id not in self.partition_map:
            self.partition_map[user_id] = user_id % self.num_partitions
        return self.partition_map[user_id]

    def get_partition_offset(self, user_id: int, total_capacity: int) -> int:
        """Calculates the memory offset for a user's retrieval region."""
        partition_id = self.assign_partition(user_id)
        partition_size = total_capacity // self.num_partitions
        return partition_id * partition_size

    def partition_batch(self, user_ids: List[int], total_capacity: int) -> torch.Tensor:
        """Returns memory offsets for a batch of users."""
        offsets = [self.get_partition_offset(uid, total_capacity) for uid in user_ids]
        return torch.tensor(offsets, dtype=torch.long)

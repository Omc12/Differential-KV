import torch
from typing import Dict, List

class RetrievalHotsetPartitioner:
    """
    Detects and partitions 'hotsets' of tokens that multiple users are 
    retrieving simultaneously, reducing bank contention on GPU.
    """
    def __init__(self, contention_threshold: int = 4):
        self.contention_threshold = contention_threshold
        self.hotsets = {}

    def detect_contention(self, user_indices: Dict[str, torch.Tensor]):
        """
        Identifies tokens being accessed by many users.
        """
        all_indices = torch.cat(list(user_indices.values()))
        counts = torch.bincount(all_indices)
        
        hot_indices = torch.where(counts >= self.contention_threshold)[0]
        return hot_indices

    def partition_hotset(self, indices: torch.Tensor, num_partitions: int = 2) -> List[torch.Tensor]:
        """
        Splits a hotset into partitions to be handled by different warps/streams.
        """
        if indices.numel() == 0:
            return []
            
        # Shuffle or split to spread the load
        perm = torch.randperm(indices.size(0))
        shuffled = indices[perm]
        
        partitions = torch.chunk(shuffled, num_partitions)
        return list(partitions)

import torch
from typing import List, Dict

class RetrievalLocalityPartitioner:
    """
    Partitions the KV cache into locality-aware zones.
    Requests are routed to the zone that contains the majority of their 
    needed sparse context, reducing global bus traffic.
    """
    def __init__(self, num_zones: int = 4):
        self.num_zones = num_zones

    def get_target_zone(self, indices: torch.Tensor, seq_len: int) -> int:
        """
        Determines the optimal zone for a set of retrieval indices.
        """
        if indices.numel() == 0:
            return 0
            
        # Divide seq_len into equal zones
        zone_size = (seq_len + self.num_zones - 1) // self.num_zones
        
        # Calculate which zone has the most hits
        zone_hits = torch.bincount(indices // zone_size, minlength=self.num_zones)
        return torch.argmax(zone_hits).item()

    def partition_request(self, request: dict, seq_len: int) -> dict:
        """Adds zone affinity to a request."""
        indices = request.get("retrieval_indices", torch.tensor([]))
        request["zone_affinity"] = self.get_target_zone(indices, seq_len)
        return request

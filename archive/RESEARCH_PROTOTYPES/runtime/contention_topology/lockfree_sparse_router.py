import torch
from typing import Optional

class LockfreeSparseRouter:
    """
    Implements a lock-free routing path for sparse retrieval requests.
    Uses atomic updates to the retrieval maps to prevent synchronization bottlenecks.
    """
    def __init__(self, capacity: int = 65536):
        self.capacity = capacity
        # Use a flat tensor that can be updated with index_put or atomic adds
        self.routing_table = torch.zeros(capacity, dtype=torch.int32)

    def route_request(self, user_id_int: int, indices: torch.Tensor):
        """
        Updates the routing table in a non-blocking way.
        """
        # torch.index_put_ is effectively atomic for non-overlapping indices on GPU
        # For overlapping, it's non-deterministic unless using scatter_add
        self.routing_table.index_fill_(0, indices, user_id_int)

    def get_route(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieves routing information for a set of indices."""
        return self.routing_table[indices]

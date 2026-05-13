import torch
from typing import Dict, Set

class RetrievalRegionIsolation:
    """
    Isolates retrieval regions for different concurrent users to prevent 
    sparse cache-line contention and interference.
    """
    def __init__(self):
        self.user_regions: Dict[str, Tuple[int, int]] = {}

    def assign_isolated_region(self, user_id: str, start: int, end: int):
        """Assigns a specific context region to a user."""
        self.user_regions[user_id] = (start, end)

    def check_interference(self, user_id: str, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Checks if the target indices interfere with other users' isolated regions.
        Returns a mask of interfering indices.
        """
        if not self.user_regions:
            return torch.zeros_like(target_indices, dtype=torch.bool)
            
        interfering = torch.zeros_like(target_indices, dtype=torch.bool)
        for uid, (start, end) in self.user_regions.items():
            if uid == user_id:
                continue
            
            # Check if any target_indices fall within (start, end)
            region_mask = (target_indices >= start) & (target_indices < end)
            interfering |= region_mask
            
        return interfering

    def get_isolation_buffer(self, user_id: str) -> int:
        """Returns the recommended buffer size between user regions."""
        return 1024 # Standard isolation window

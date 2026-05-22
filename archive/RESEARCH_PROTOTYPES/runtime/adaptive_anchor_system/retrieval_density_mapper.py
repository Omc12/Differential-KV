import torch
import numpy as np
from typing import Dict, List, Optional

class RetrievalDensityMapper:
    """
    Maps retrieval density across the context window to identify 'sparse collapse' regions.
    Used to drive adaptive anchor spacing.
    """
    def __init__(self, window_size: int = 1024, decay: float = 0.99):
        self.window_size = window_size
        self.decay = decay
        self.density_map: Optional[torch.Tensor] = None
        self.collision_counts: Optional[torch.Tensor] = None

    def update(self, retrieval_indices: torch.Tensor, success_mask: torch.Tensor, total_seq_len: int):
        """
        Updates the density map based on retrieval performance.
        retrieval_indices: [N] indices being retrieved
        success_mask: [N] boolean mask of whether retrieval was successful
        """
        if self.density_map is None or self.density_map.size(0) < total_seq_len:
            new_map = torch.ones(total_seq_len, device=retrieval_indices.device)
            new_collisions = torch.zeros(total_seq_len, device=retrieval_indices.device)
            if self.density_map is not None:
                new_map[:self.density_map.size(0)] = self.density_map
                new_collisions[:self.collision_counts.size(0)] = self.collision_counts
            self.density_map = new_map
            self.collision_counts = new_collisions

        # Apply decay to existing map
        self.density_map *= self.decay

        # Update map with new retrieval data
        # We use scatter_add for atomic-like updates in GPU-friendly way if needed, 
        # but here we just update the specific indices.
        success_weights = success_mask.float()
        failure_weights = (1.0 - success_weights) * 2.0 # Penalize failures more
        
        self.density_map.index_add_(0, retrieval_indices, -failure_weights)
        self.density_map.index_add_(0, retrieval_indices, success_weights * 0.1)
        
        # Clip to [0, 1] range roughly (using sigmoid or clamp)
        self.density_map = torch.clamp(self.density_map, 0.0, 1.0)

    def get_region_density(self, start: int, end: int) -> float:
        """Returns the average retrieval density for a specific region."""
        if self.density_map is None:
            return 1.0
        region = self.density_map[start:end]
        if region.numel() == 0:
            return 1.0
        return region.mean().item()

    def identify_hotspots(self, threshold: float = 0.3) -> torch.Tensor:
        """Returns indices where retrieval density is dangerously low."""
        if self.density_map is None:
            return torch.tensor([], dtype=torch.long)
        return torch.where(self.density_map < threshold)[0]

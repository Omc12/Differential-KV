import torch
import numpy as np
from typing import List

class KernelOccupancyHeatmaps:
    """
    PHASE 7.5B: Kernel Occupancy Heatmaps
    Visualizes the distribution of sparse kernel workload across 
    GPU Streaming Multiprocessors (SMs).
    """
    def __init__(self, num_sm: int = 108): # Default for A100/H100
        self.num_sm = num_sm
        self.occupancy_history: List[np.ndarray] = []

    def update_occupancy(self, grid_dim: tuple, block_dim: tuple):
        """
        Estimates SM occupancy based on grid distribution.
        """
        # Simplified occupancy mapping: which SMs are likely busy
        # In a real tool, we'd use CUPTI or similar to get real SM IDs
        sm_map = np.zeros(self.num_sm)
        total_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
        
        # Round-robin block distribution simulation
        for i in range(total_blocks):
            sm_map[i % self.num_sm] += 1
            
        # Normalize to 0-1
        if sm_map.max() > 0:
            sm_map /= sm_map.max()
            
        self.occupancy_history.append(sm_map)

    def get_current_heatmap(self) -> np.ndarray:
        """Returns the latest occupancy map."""
        return self.occupancy_history[-1] if self.occupancy_history else np.zeros(self.num_sm)

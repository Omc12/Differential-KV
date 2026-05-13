import torch
import numpy as np

class KernelOccupancyHeatmaps:
    """
    Generates data for occupancy heatmaps, showing which SMs are active 
    during sparse kernel execution.
    """
    def __init__(self, num_sm: int = 108): # A100 default
        self.num_sm = num_sm
        self.sm_activity = torch.zeros(num_sm)

    def record_launch(self, grid: tuple, threads_per_block: int):
        """
        Estimates which SMs are hit by the launch.
        """
        num_blocks = grid[0] * grid[1] * grid[2]
        # Simplistic mapping: blocks distributed over SMs
        for i in range(num_blocks):
            self.sm_activity[i % self.num_sm] += 1

    def get_heatmap_data(self) -> np.ndarray:
        return self.sm_activity.cpu().numpy()

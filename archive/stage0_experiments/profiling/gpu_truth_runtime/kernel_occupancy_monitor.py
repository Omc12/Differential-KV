import torch

class KernelOccupancyMonitor:
    """
    Estimates GPU kernel occupancy for sparse operations.
    Helps identify warp divergence and resource underutilization.
    """
    def __init__(self, device_id: int = 0):
        self.props = torch.cuda.get_device_properties(device_id)
        self.max_threads_per_sm = self.props.max_threads_per_multi_processor
        self.num_sm = self.props.multi_processor_count

    def estimate_occupancy(self, threads_per_block: int, blocks_per_grid: int) -> float:
        """
        Estimates theoretical occupancy.
        Occupancy = Active Warps / Max Warps
        """
        total_threads = threads_per_block * blocks_per_grid
        max_threads = self.max_threads_per_sm * self.num_sm
        
        theoretical_occupancy = min(1.0, total_threads / max_threads)
        return theoretical_occupancy

    def track_warp_divergence(self, sparse_indices: torch.Tensor, block_size: int = 32) -> float:
        """
        Heuristic for warp divergence: checks how 'ragged' or 'sparse' the indices are
        within a warp-sized block.
        """
        if sparse_indices.numel() == 0:
            return 0.0
            
        # If indices are sequential, divergence is low.
        # If indices are scattered, divergence is high.
        diffs = sparse_indices[1:] - sparse_indices[:-1]
        non_sequential = (diffs != 1).float().mean().item()
        
        return non_sequential

    def get_hardware_constraints(self) -> dict:
        """Returns hardware constraints for kernel optimization."""
        return {
            "sm_count": self.num_sm,
            "max_threads_per_sm": self.max_threads_per_sm,
            "warp_size": 32,
            "shared_mem_per_sm_kb": self.props.shared_memory_per_block / 1024
        }

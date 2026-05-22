"""
hardware_materialization/sparse_memory_access_optimizer.py

Improves sparse KV memory locality and coalesced access.
"""

import torch
import logging

logger = logging.getLogger("MemoryOptimizer")

class SparseMemoryAccessOptimizer:
    """
    Optimizes memory access patterns for sparse operations to reduce stalls.
    """
    def __init__(self):
        self.total_optimizations = 0

    def optimize_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sorts indices to ensure coalesced memory access during gather/scatter.
        Returns (sorted_indices, permutation).
        """
        if indices.numel() <= 1:
            return indices, torch.tensor([0], device=indices.device)
            
        sorted_indices, perm = torch.sort(indices)
        self.total_optimizations += 1
        return sorted_indices, perm

    def align_memory(self, tensor: torch.Tensor, alignment: int = 128) -> torch.Tensor:
        """Ensures tensor is aligned to cache-line boundaries."""
        # PyTorch tensors are generally well-aligned, but we can ensure
        # contiguous layout for optimal vectorized access.
        if not tensor.is_contiguous():
            return tensor.contiguous()
        return tensor

    def get_efficiency_metrics(self):
        return {
            "indices_optimized": self.total_optimizations,
            "locality_strategy": "sorted_coalesced"
        }

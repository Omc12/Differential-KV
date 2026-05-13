"""
memory/warp_locality_optimizer.py

Optimizes KV memory layout for warp-level locality.
Reduces warp divergence by ensuring threads in a warp access adjacent memory.
"""

import torch

class WarpLocalityOptimizer:
    def __init__(self, warp_size: int = 32):
        self.warp_size = warp_size
        self.divergence_metrics = []

    def optimize_warp_layout(self, indices: torch.Tensor):
        """
        Groups indices into warp-sized blocks to minimize memory transactions.
        """
        # Logic: Sort indices within each warp block to ensure coalesced access
        num_indices = indices.size(0)
        num_warps = (num_indices + self.warp_size - 1) // self.warp_size
        
        optimized = indices.clone()
        for i in range(num_warps):
            start = i * self.warp_size
            end = min((i + 1) * self.warp_size, num_indices)
            # Sorting within the warp block to improve DRAM coalescing
            optimized[start:end], _ = torch.sort(optimized[start:end])
            
        return optimized

    def calculate_warp_efficiency(self, indices: torch.Tensor):
        """
        Estimates warp efficiency based on index span within each warp block.
        High efficiency = low span (coalesced access).
        """
        num_indices = indices.size(0)
        num_warps = (num_indices + self.warp_size - 1) // self.warp_size
        
        spans = []
        for i in range(num_warps):
            start = i * self.warp_size
            end = min((i + 1) * self.warp_size, num_indices)
            warp_indices = indices[start:end]
            span = (torch.max(warp_indices) - torch.min(warp_indices)).item()
            spans.append(span)
            
        avg_span = sum(spans) / len(spans) if spans else 0
        efficiency = 1.0 / (1.0 + avg_span / 128.0) # Heuristic
        self.divergence_metrics.append(efficiency)
        return efficiency

    def get_warp_report(self):
        """Returns average warp efficiency and divergence trends."""
        if not self.divergence_metrics: return {}
        return {
            "avg_warp_efficiency": sum(self.divergence_metrics) / len(self.divergence_metrics),
            "peak_warp_efficiency": max(self.divergence_metrics)
        }

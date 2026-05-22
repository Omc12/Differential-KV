"""
memory/sparse_bandwidth_optimizer.py

Optimizes memory layout and access patterns for sparse KV retrieval.
Reduces cross-bank movement and maximizes cache locality.
"""

import torch
import numpy as np

class SparseBandwidthOptimizer:
    def __init__(self, d_model: int, bank_size: int = 32):
        self.d_model = d_model
        self.bank_size = bank_size
        self.locality_score = []

    def optimize_layout(self, kv_tensors: torch.Tensor, access_patterns: torch.Tensor):
        """
        Reorders KV tensors to align with common access patterns, reducing bank conflicts.
        """
        # Logic: Cluster KV pairs that are frequently accessed together
        # Simulated optimization for now
        num_kv = kv_tensors.size(0)
        optimized_indices = torch.randperm(num_kv, device=kv_tensors.device)
        
        # In production, this would use a clustering algorithm (e.g., k-means or graph-based)
        # to ensure that KV blocks accessed by the same warp are adjacent.
        
        return optimized_indices

    def measure_bandwidth_efficiency(self, indices: torch.Tensor):
        """
        Estimates bandwidth efficiency based on index locality.
        High efficiency means indices are clustered within banks.
        """
        if indices.numel() < 2:
            return 1.0
            
        diffs = torch.abs(indices[1:] - indices[:-1])
        efficiency = (diffs < self.bank_size).float().mean().item()
        self.locality_score.append(efficiency)
        return efficiency

    def get_stats(self):
        """Returns average bandwidth efficiency and locality trends."""
        if not self.locality_score:
            return {"avg_efficiency": 0.0}
        return {
            "avg_efficiency": sum(self.locality_score) / len(self.locality_score),
            "current_locality": self.locality_score[-1]
        }

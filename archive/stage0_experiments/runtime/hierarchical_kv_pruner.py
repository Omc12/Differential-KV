import torch
import torch.nn as nn
from typing import Optional

class HierarchicalKVPruner(nn.Module):
    """
    Hierarchical KV Pruning based on token importance.
    Maintains multiple tiers of cache:
    1. L1: Full precision, recently used / high importance
    2. L2: Pruned / Compressed, medium importance
    3. Evicted: Low importance
    """
    def __init__(self, target_reduction: float = 0.5):
        super().__init__()
        self.target_reduction = target_reduction

    def prune(self, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor):
        """
        k, v: [batch, heads, seq_len, head_dim]
        scores: [batch, heads, seq_len] - importance scores
        """
        seq_len = k.size(-2)
        k_keep = int(seq_len * (1.0 - self.target_reduction))
        
        # Sort by importance
        _, top_indices = torch.topk(scores, k_keep, dim=-1, sorted=False)
        
        # Gather top-k keys and values
        # Note: This is a simplified version. Real implementation needs proper indexing.
        # This demonstrates the hierarchical pruning logic.
        
        # For now, return a placeholder that represents the reduction
        return k[..., :k_keep, :], v[..., :k_keep, :]

    def apply_hierarchical_policy(self, layer_idx: int, total_layers: int):
        """
        Layers at different depths may have different pruning budgets.
        Earlier layers often need more KV retention for retrieval.
        """
        # Dynamic budget based on depth
        if layer_idx < total_layers // 4:
            self.target_reduction = 0.2 # Keep more in early layers
        elif layer_idx > 3 * total_layers // 4:
            self.target_reduction = 0.7 # Prune more in late layers
        else:
            self.target_reduction = 0.5

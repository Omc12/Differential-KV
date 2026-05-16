"""
anchor_logic/learned_rank_scheduler.py
Phase 16: Learned Dynamic Rank Allocation
Self-optimizing cognitive memory allocation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

class LearnedRankScheduler(nn.Module):
    def __init__(self, input_dim: int = 12, num_layers: int = 12, max_rank: int = 32):
        super().__init__()
        self.max_rank = max_rank
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers) # Per-layer rank factor
        )
        
        # Head-wise sensitivity head (simplified)
        self.head_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8) # Assuming 8 heads for simplicity
        )

    def forward(self, metrics_tensor: torch.Tensor):
        """
        Input: Metrics from Guard.prepare_input
        Output: Per-layer and per-head rank factors [0, 1]
        """
        layer_factors = torch.sigmoid(self.net(metrics_tensor))
        head_factors = torch.sigmoid(self.head_net(metrics_tensor))
        
        return {
            "layer_ranks": (layer_factors * self.max_rank).round().int(),
            "head_importance": head_factors
        }

class AdaptiveRankController:
    def __init__(self, scheduler: LearnedRankScheduler):
        self.scheduler = scheduler

    def get_ranks(self, metrics: Dict[str, float], pos: int, max_pos: int) -> Dict[str, Any]:
        # Using the same input preparation as Guard for consistency
        from anchor_logic.cognitive_guard_network import CognitiveGuardNetwork
        input_tensor = CognitiveGuardNetwork.prepare_input(metrics, pos, max_pos, 0, 0).unsqueeze(0)
        
        with torch.no_grad():
            alloc = self.scheduler(input_tensor)
            
        return {
            "layer_ranks": alloc["layer_ranks"][0].tolist(),
            "head_importance": alloc["head_importance"][0].tolist()
        }

if __name__ == "__main__":
    scheduler = LearnedRankScheduler()
    metrics = {"latent_velocity": 0.2, "hidden_drift": 0.05}
    controller = AdaptiveRankController(scheduler)
    ranks = controller.get_ranks(metrics, 50, 100)
    print("Allocated Ranks:", ranks)

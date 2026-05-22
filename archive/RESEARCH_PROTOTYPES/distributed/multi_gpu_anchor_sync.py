"""
distributed/multi_gpu_anchor_sync.py

Synchronizes sparse KV anchors across multiple GPU devices.
Uses torch.distributed for low-latency synchronization.
"""

import torch
import torch.distributed as dist
from typing import List, Optional
import logging

class MultiGPUAnchorSync:
    """
    Handles synchronization of anchor states and importance scores 
    across distributed processes.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.logger = logging.getLogger("MultiGPUAnchorSync")
        self.is_distributed = dist.is_available() and dist.is_initialized()

    def sync_anchors(self, anchor_tensors: torch.Tensor):
        """
        Broadcasts or reduces anchor states across all processes.
        """
        if not self.is_distributed:
            return anchor_tensors

        # In a real sparse setup, we only sync the most important anchors
        # to save bandwidth.
        dist.all_reduce(anchor_tensors, op=dist.ReduceOp.SUM)
        # Average the values
        anchor_tensors /= dist.get_world_size()
        
        return anchor_tensors

    def sync_importance_scores(self, scores: torch.Tensor):
        """
        Synchronizes importance scores to ensure consistent pruning 
        decisions across the cluster.
        """
        if not self.is_distributed:
            return scores

        # All-gather scores to make global pruning decisions
        world_size = dist.get_world_size()
        gathered_scores = [torch.zeros_like(scores) for _ in range(world_size)]
        dist.all_gather(gathered_scores, scores)
        
        return torch.cat(gathered_scores, dim=0)

    def broadcast_policy(self, policy_params: torch.Tensor, src_rank: int = 0):
        """
        Broadcasts optimization policy parameters from a leader node.
        """
        if not self.is_distributed:
            return policy_params
            
        dist.broadcast(policy_params, src=src_rank)
        return policy_params

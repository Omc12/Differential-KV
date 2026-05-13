"""
distributed/distributed_sparse_router.py

Core routing logic for directing retrieval queries to the correct 
sharded sparse KV node in a distributed environment.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
import logging

class DistributedSparseRouter:
    """
    Routes sparse retrieval queries across multiple nodes based on 
    shard ownership and retrieval locality.
    """
    def __init__(self, n_nodes: int, shard_manager: Any):
        self.n_nodes = n_nodes
        self.shard_manager = shard_manager
        self.logger = logging.getLogger("DistributedSparseRouter")

    def route_query(self, query: torch.Tensor, sequence_index: int) -> int:
        """
        Determines which node should handle a query for a specific sequence index.
        In Differential KV, sequence indices are mapped to sparse shards.
        """
        # 1. Check Shard Manager for current ownership
        target_node = self.shard_manager.get_owner(sequence_index)
        
        if target_node is None:
            # 2. Fallback to consistent hashing if shard is not yet allocated
            target_node = sequence_index % self.n_nodes
            self.logger.warning(f"Shard for index {sequence_index} unallocated, defaulting to Node {target_node}")
            
        return target_node

    def route_batch(self, batch_queries: torch.Tensor, batch_indices: List[int]) -> Dict[int, List[int]]:
        """
        Routes a batch of queries, returning a mapping of Node ID to batch indices.
        """
        routing_map = {}
        for i, seq_idx in enumerate(batch_indices):
            node_id = self.route_query(batch_queries[i], seq_idx)
            if node_id not in routing_map:
                routing_map[node_id] = []
            routing_map[node_id].append(i)
            
        return routing_map

    def optimize_locality(self, access_patterns: Dict[int, int]):
        """
        Informs the shard manager about access frequencies to optimize locality.
        """
        for seq_idx, freq in access_patterns.items():
            self.shard_manager.report_access(seq_idx, freq)

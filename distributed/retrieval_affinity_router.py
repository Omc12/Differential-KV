"""
distributed/retrieval_affinity_router.py

Routes queries with high affinity to where their anchors already reside.
Minimizes 'cold' starts for retrieval operations.
"""

from typing import Dict, List, Any
import logging

class RetrievalAffinityRouter:
    """
    Stateful router that tracks anchor residency and caches affinity.
    """
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.residency_map: Dict[int, List[int]] = {} # shard_id -> list of nodes
        self.logger = logging.getLogger("RetrievalAffinityRouter")

    def register_residency(self, shard_id: int, node_id: int):
        """Updates the router on where a shard is currently cached."""
        if shard_id not in self.residency_map:
            self.residency_map[shard_id] = []
        if node_id not in self.residency_map[shard_id]:
            self.residency_map[shard_id].append(node_id)

    def route_with_affinity(self, shard_id: int, fallback_node: int) -> int:
        """
        Routes to a node that already has the shard resident.
        """
        resident_nodes = self.residency_map.get(shard_id, [])
        if resident_nodes:
            # Pick first resident node for simplicity
            return resident_nodes[0]
            
        return fallback_node

    def evict_residency(self, shard_id: int, node_id: int):
        """Removes a node from the residency list for a shard."""
        if shard_id in self.residency_map and node_id in self.residency_map[shard_id]:
            self.residency_map[shard_id].remove(node_id)

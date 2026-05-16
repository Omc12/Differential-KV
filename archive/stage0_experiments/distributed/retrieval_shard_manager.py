"""
distributed/retrieval_shard_manager.py

Manages shard ownership and locality across the distributed cluster.
Tracks where each sparse KV block lives and handles migrations.
"""

from typing import Dict, List, Optional, Any
import time
import logging

class RetrievalShardManager:
    """
    Stateful manager for KV shard distribution.
    """
    def __init__(self, n_nodes: int, shard_size: int = 1024):
        self.n_nodes = n_nodes
        self.shard_size = shard_size
        self.shard_to_node: Dict[int, int] = {} # shard_id -> node_id
        self.node_to_shards: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        self.access_counts: Dict[int, int] = {}
        self.last_migration = time.time()
        self.logger = logging.getLogger("RetrievalShardManager")

    def get_owner(self, sequence_index: int) -> Optional[int]:
        """Returns the node ID owning the shard containing sequence_index."""
        shard_id = sequence_index // self.shard_size
        return self.shard_to_node.get(shard_id)

    def allocate_shard(self, shard_id: int, node_id: int):
        """Explicitly allocates a shard to a node."""
        if shard_id in self.shard_to_node:
            old_node = self.shard_to_node[shard_id]
            self.node_to_shards[old_node].remove(shard_id)
            
        self.shard_to_node[shard_id] = node_id
        self.node_to_shards[node_id].append(shard_id)
        self.logger.info(f"Shard {shard_id} allocated to Node {node_id}")

    def report_access(self, sequence_index: int, frequency: int = 1):
        """Tracks access patterns for future migration decisions."""
        shard_id = sequence_index // self.shard_size
        self.access_counts[shard_id] = self.access_counts.get(shard_id, 0) + frequency

    def rebalance(self):
        """
        Periodically rebalances shards based on access density.
        Shards with high access frequency should be distributed to avoid hotspots.
        """
        # REAL implementation would use a load-balancing algorithm
        self.logger.info("Triggering shard rebalance...")
        # Placeholder for real migration logic
        self.last_migration = time.time()

    def get_cluster_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the cluster sharding state."""
        return {
            "n_nodes": self.n_nodes,
            "shard_count": len(self.shard_to_node),
            "distribution": {node: len(shards) for node, shards in self.node_to_shards.items()},
            "hotspots": sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

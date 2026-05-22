"""
distributed/hierarchical_anchor_residency.py

Manages anchor residency in a tiered hierarchy (Local GPU -> Remote GPU -> Disk).
Optimizes for retrieval latency vs memory capacity.
"""

from typing import Dict, Any, List, Optional
import logging

class HierarchicalAnchorResidency:
    """
    Tiered residency manager for sparse anchors.
    """
    def __init__(self, local_capacity: int, remote_capacity: int):
        self.local_capacity = local_capacity
        self.remote_capacity = remote_capacity
        self.local_shards: List[int] = []
        self.remote_shards: List[int] = []
        self.disk_shards: List[int] = []
        self.logger = logging.getLogger("HierarchicalAnchorResidency")

    def access_shard(self, shard_id: int):
        """
        Moves a shard up the hierarchy if it's frequently accessed.
        """
        if shard_id in self.local_shards:
            # Promote to front of MRU
            self.local_shards.remove(shard_id)
            self.local_shards.append(shard_id)
        elif shard_id in self.remote_shards:
            self._promote_to_local(shard_id)
        elif shard_id in self.disk_shards:
            self._promote_to_remote(shard_id)
        else:
            self._add_to_disk(shard_id)

    def _promote_to_local(self, shard_id: int):
        """Moves from remote GPU to local GPU memory."""
        if len(self.local_shards) >= self.local_capacity:
            evicted = self.local_shards.pop(0)
            self.remote_shards.append(evicted)
        
        if shard_id in self.remote_shards:
            self.remote_shards.remove(shard_id)
        self.local_shards.append(shard_id)

    def _promote_to_remote(self, shard_id: int):
        """Moves from disk to remote GPU memory."""
        if len(self.remote_shards) >= self.remote_capacity:
            evicted = self.remote_shards.pop(0)
            self.disk_shards.append(evicted)
            
        if shard_id in self.disk_shards:
            self.disk_shards.remove(shard_id)
        self.remote_shards.append(shard_id)

    def _add_to_disk(self, shard_id: int):
        """Initial placement on disk."""
        self.disk_shards.append(shard_id)

    def get_residency_summary(self) -> Dict[str, int]:
        """Returns counts for each tier."""
        return {
            "local": len(self.local_shards),
            "remote": len(self.remote_shards),
            "disk": len(self.disk_shards)
        }

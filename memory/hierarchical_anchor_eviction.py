"""
memory/hierarchical_anchor_eviction.py

Advanced eviction logic based on retrieval frequency and context depth.
Protects 'essential' anchors even under extreme memory pressure.
"""

from typing import Dict, List, Set, Any
import logging

class HierarchicalAnchorEviction:
    """
    Priority-aware eviction manager for sparse anchors.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.anchors: Dict[int, float] = {} # shard_id -> importance_score
        self.protected: Set[int] = set() # Sink tokens, system anchors
        self.logger = logging.getLogger("HierarchicalAnchorEviction")

    def register_anchor(self, shard_id: int, score: float, is_protected: bool = False):
        """Adds or updates an anchor in the managed set."""
        self.anchors[shard_id] = score
        if is_protected:
            self.protected.add(shard_id)
            
        if len(self.anchors) > self.capacity:
            self.evict_low_priority()

    def evict_low_priority(self):
        """Evicts the least important non-protected anchors."""
        # Filter out protected ones
        candidates = {sid: score for sid, score in self.anchors.items() if sid not in self.protected}
        
        if not candidates:
            self.logger.warning("MEM_PRESSURE: No non-protected anchors to evict!")
            return

        # Evict anchor with lowest score
        to_evict = min(candidates, key=candidates.get)
        del self.anchors[to_evict]
        self.logger.info(f"Evicted Anchor {to_evict} (Score: {candidates[to_evict]:.4f})")

    def set_protected(self, shard_id: int, protect: bool = True):
        """Toggles protection for a specific shard."""
        if protect:
            self.protected.add(shard_id)
        elif shard_id in self.protected:
            self.protected.remove(shard_id)

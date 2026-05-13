"""
memory/retrieval_aging_tracker.py

Tracks and degrades retrieval scores for regions that haven't been 
accessed in long-horizon sessions.
"""

import time
from typing import Dict, List
import logging

class RetrievalAgingTracker:
    """
    Manages temporal decay of sparse retrieval anchors.
    """
    def __init__(self, half_life_seconds: float = 3600.0): # 1 hour default
        self.half_life = half_life_seconds
        self.last_access: Dict[int, float] = {}
        self.logger = logging.getLogger("RetrievalAgingTracker")

    def record_access(self, shard_id: int):
        """Records an access event for a shard."""
        self.last_access[shard_id] = time.time()

    def get_age_multiplier(self, shard_id: int) -> float:
        """
        Calculates a decay multiplier [0, 1] based on time since last access.
        Uses exponential decay.
        """
        if shard_id not in self.last_access:
            return 0.0
            
        dt = time.time() - self.last_access[shard_id]
        # multiplier = 0.5 ^ (dt / half_life)
        return 0.5 ** (dt / self.half_life)

    def prune_stale_records(self, max_age: float = 86400.0):
        """Removes records older than one day."""
        now = time.time()
        stale = [sid for sid, ts in self.last_access.items() if now - ts > max_age]
        for sid in stale:
            del self.last_access[sid]
        
        if stale:
            self.logger.info(f"Pruned {len(stale)} stale retrieval records.")

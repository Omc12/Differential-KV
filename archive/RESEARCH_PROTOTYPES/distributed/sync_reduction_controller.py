"""
distributed/sync_reduction_controller.py

Manages and reduces the frequency of cross-node synchronizations.
Uses lazy synchronization and adaptive consistency boundaries.
"""

import time
import logging

class SyncReductionController:
    """
    Controller for distributed synchronization pressure.
    """
    def __init__(self, min_sync_interval_ms: float = 50.0):
        self.min_sync_interval = min_sync_interval_ms / 1000.0
        self.last_sync = 0.0
        self.sync_count = 0
        self.logger = logging.getLogger("SyncReductionController")

    def should_sync(self, urgency: float = 1.0) -> bool:
        """
        Determines if a synchronization event should occur now.
        Higher urgency (e.g. KV eviction required) bypasses the interval.
        """
        now = time.time()
        time_since = now - self.last_sync
        
        # 1. Respect minimum interval unless urgent
        if time_since < self.min_sync_interval and urgency < 0.9:
            return False
            
        # 2. Perform Sync
        self.last_sync = now
        self.sync_count += 1
        return True

    def get_sync_stats(self) -> dict:
        """Returns stats on synchronization frequency and reduction."""
        return {
            "total_syncs": self.sync_count,
            "avg_sync_interval_ms": (time.time() - self.last_sync) * 1000,
            "suppressed_syncs": 150 # Placeholder for reduction tracking
        }

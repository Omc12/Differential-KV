"""
distributed/sync_pressure_patch.py

Patches synchronization logic to reduce bursts and spikes.
Implements jitter-aware synchronization scheduling.
"""

import time
import random
import logging

class SyncPressurePatch:
    """
    Patch for smoothing out synchronization pressure.
    """
    def __init__(self, base_interval_ms: float = 50.0):
        self.base_interval = base_interval_ms / 1000.0
        self.last_sync = 0.0
        self.logger = logging.getLogger("SyncPressurePatch")

    def get_patched_sync_wait(self) -> float:
        """
        Calculates a jittered wait time to prevent 'thundering herd' syncs.
        """
        jitter = random.uniform(0.9, 1.1)
        target_interval = self.base_interval * jitter
        
        now = time.time()
        time_since = now - self.last_sync
        
        wait = max(0.0, target_interval - time_since)
        self.last_sync = now + wait
        return wait

    def apply_pressure_relief(self):
        """Intentionally slows down synchronization if queue pressure is high."""
        self.logger.info("Applying synchronization pressure relief...")
        time.sleep(0.005) # Small stabilization delay

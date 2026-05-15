"""
hardware_materialization/deterministic_microbatch_controller.py

Maintains replay-safe, deterministic microbatch scheduling for Differential KV.
"""

import logging
from typing import List, Any

logger = logging.getLogger("MicrobatchController")

class DeterministicMicrobatchController:
    """
    Enforces stable batch ordering and decode grouping to ensure replay consistency.
    """
    def __init__(self):
        self.locked_batch_size = 1
        self.is_locked = False

    def lock_microbatch_size(self, size: int):
        """Locks the microbatch size to ensure deterministic replay."""
        self.locked_batch_size = size
        self.is_locked = True
        logger.info(f"Microbatch size locked to {size} for deterministic replay.")

    def unlock(self):
        self.is_locked = False
        logger.info("Microbatch size unlocked.")

    def get_batch_indices(self, total_count: int) -> List[List[int]]:
        """
        Returns a deterministic sequence of indices for microbatching.
        """
        batch_size = self.locked_batch_size
        indices = list(range(total_count))
        
        batches = []
        for i in range(0, total_count, batch_size):
            batches.append(indices[i:i + batch_size])
            
        return batches

    def verify_consistency(self, batch_a: List[int], batch_b: List[int]) -> bool:
        """Verifies if two batchings are identical."""
        return batch_a == batch_b

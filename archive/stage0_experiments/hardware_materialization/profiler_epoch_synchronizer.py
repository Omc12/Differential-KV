"""
hardware_materialization/profiler_epoch_synchronizer.py

Synchronizes profiler telemetry with tuning state transitions.
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger("EpochSync")

class ProfilerEpochSynchronizer:
    """
    Manages epoch-based telemetry to ensure snapshots align with tuning updates.
    """
    def __init__(self):
        self.current_epoch = 0
        self.epoch_timestamps: Dict[int, float] = {}
        self.tuning_states: Dict[int, Any] = {}

    def start_epoch(self, tuning_state: Any):
        """Starts a new synchronized telemetry epoch."""
        self.current_epoch += 1
        self.epoch_timestamps[self.current_epoch] = time.time()
        self.tuning_states[self.current_epoch] = tuning_state
        logger.info(f"Starting telemetry epoch {self.current_epoch} with tuning state: {tuning_state}")

    def get_epoch_alignment(self, timestamp: float) -> int:
        """Finds the epoch that corresponds to a given telemetry timestamp."""
        best_epoch = 0
        for epoch, ts in self.epoch_timestamps.items():
            if ts <= timestamp:
                best_epoch = epoch
        return best_epoch

    def verify_alignment(self, trace_epoch: int) -> bool:
        """Verifies if a trace was captured in the current tuning epoch."""
        return trace_epoch == self.current_epoch

"""
hardware_materialization/occupancy_hysteresis_stabilizer.py

Prevents occupancy instability and oscillation caused by aggressive retuning.
"""

import logging
import time
from typing import Dict

logger = logging.getLogger("OccupancyHysteresis")

class OccupancyHysteresisStabilizer:
    """
    Enforces cooldown windows and stabilization thresholds for kernel retuning.
    """
    def __init__(self, cooldown_seconds: float = 5.0, threshold: float = 0.05):
        self.cooldown = cooldown_seconds
        self.threshold = threshold
        self.last_tuning_time: Dict[str, float] = {}
        self.last_occupancy: Dict[str, float] = {}

    def should_retune(self, key: str, current_occupancy: float) -> bool:
        """
        Determines if a kernel should be retuned based on hysteresis and cooldown.
        """
        now = time.time()
        
        # Check cooldown
        if key in self.last_tuning_time:
            if now - self.last_tuning_time[key] < self.cooldown:
                return False
                
        # Check threshold (hysteresis)
        if key in self.last_occupancy:
            delta = abs(current_occupancy - self.last_occupancy[key])
            if delta < self.threshold:
                # Occupancy shift is too small to justify retuning
                return False
                
        return True

    def record_tuning(self, key: str, occupancy: float):
        """Records a tuning event to reset cooldown."""
        self.last_tuning_time[key] = time.time()
        self.last_occupancy[key] = occupancy
        logger.info(f"Occupancy tuning recorded for '{key}': {occupancy:.4f}")

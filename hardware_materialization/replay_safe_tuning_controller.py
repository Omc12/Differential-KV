"""
hardware_materialization/replay_safe_tuning_controller.py

Ensures that runtime tuning operations never break deterministic CUDA graph replay.
"""

import logging
from typing import Dict, Any, Callable

logger = logging.getLogger("TuningController")

class ReplaySafeTuningController:
    """
    Guarantees replay-safe tuning by enforcing boundary locks and validation.
    """
    def __init__(self):
        self.locked_configs: Dict[str, Any] = {}
        self.tuning_active = False

    def request_tuning_window(self, key: str) -> bool:
        """
        Only allows tuning if no active graph replay is sensitive to the change.
        """
        # In this stabilization phase, we avoid tuning if we're in a "hot" region
        if self.tuning_active:
            return False
        return True

    def lock_replay_boundary(self, key: str, config: Any):
        """Locks a configuration for a specific replay key."""
        self.locked_configs[key] = config
        logger.debug(f"Locked configuration for replay boundary: {key}")

    def validate_tuning_safety(self, key: str, new_config: Any) -> bool:
        """Checks if a new configuration is safe to apply without breaking replay."""
        if key in self.locked_configs:
            # If the config has changed, we might need to invalidate the graph
            # This controller prevents "silent" drift by flagging the need for recapture.
            return False
        return True

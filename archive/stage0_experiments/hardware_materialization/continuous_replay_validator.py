"""
hardware_materialization/continuous_replay_validator.py

Validates CUDA graph replay consistency over extended serving runs.
"""

import torch
import logging
from typing import Dict

logger = logging.getLogger("ReplayValidator")

class ContinuousReplayValidator:
    """
    Detects delayed nondeterministic drift during sustained serving.
    """
    def __init__(self):
        self.reference_hashes: Dict[str, int] = {}
        self.drift_events = 0

    def capture_hash(self, key: str, output: torch.Tensor):
        """Captures a checksum of the output to detect later drift."""
        # Simple checksum for deterministic verification
        h = hash(tuple(output.view(-1)[:100].tolist()))
        self.reference_hashes[key] = h

    def validate_replay(self, key: str, current_output: torch.Tensor) -> bool:
        """
        Compares current replay output against the reference hash.
        """
        if key not in self.reference_hashes:
            self.capture_hash(key, current_output)
            return True
            
        current_h = hash(tuple(current_output.view(-1)[:100].tolist()))
        if current_h != self.reference_hashes[key]:
            logger.error(f"Replay drift detected for '{key}'! Determinism compromised.")
            self.drift_events += 1
            return False
            
        return True

    def get_drift_score(self) -> float:
        """Returns 0.0 if no drift, >0.0 if drift occurred."""
        return float(self.drift_events)

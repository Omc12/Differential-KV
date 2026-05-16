import os
from typing import Dict, Any

class SparseParticipationController:
    """
    Manages global sparse participation limits and pressure.
    Enforces DIFFKV_AGGRESSIVE_SPARSE_MODE.
    """
    def __init__(self):
        self.aggressive_mode = os.environ.get("DIFFKV_AGGRESSIVE_SPARSE_MODE") == "1"
        self.config = {
            "max_participation_ratio": 0.05 if self.aggressive_mode else 0.2,
            "eviction_pressure": 0.9 if self.aggressive_mode else 0.5,
            "disable_conservative_fallbacks": self.aggressive_mode,
            "min_dense_recovery": not self.aggressive_mode
        }

    def get_participation_limit(self, seq_len: int) -> int:
        """
        Calculates hard participation limit for a given sequence length.
        """
        limit = int(seq_len * self.config["max_participation_ratio"])
        return max(1, limit)

    def should_bypass_dense_recovery(self) -> bool:
        """
        In aggressive mode, we minimize dense recovery paths.
        """
        return self.config["disable_conservative_fallbacks"]

    def get_config(self) -> Dict[str, Any]:
        return self.config

controller = SparseParticipationController()

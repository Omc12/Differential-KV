"""
STAGE 2.5 - SRC: Operational Envelope Mapper
Phase 40.0 - Scientific Research Consolidation

Produces explicit operational maps for safe sparse ratios and contexts.
"""
import threading
from typing import Dict, Any

class OperationalEnvelopeMapper:
    def __init__(self):
        self._lock = threading.RLock()
        self._max_safe_ratio = 1.0
        self._min_safe_ratio = 0.0

    def update_envelope(self, sparse_ratio: float, fidelity: float):
        with self._lock:
            if fidelity > 0.8:
                if sparse_ratio < self._max_safe_ratio:
                    self._max_safe_ratio = sparse_ratio
            else:
                if sparse_ratio > self._min_safe_ratio:
                    self._min_safe_ratio = sparse_ratio

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "safe_ratio_upper_bound": round(self._max_safe_ratio, 4),
                "safe_ratio_lower_bound": round(self._min_safe_ratio, 4)
            }

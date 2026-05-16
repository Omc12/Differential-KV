"""
STAGE 2 - HSZ: Dense Criticality Detector
Phase 39.3 - Hybrid Semantic Zoning

Detects which layers/heads show high semantic sensitivity to sparse execution.
Measures collapse sensitivity based on drift spikes, fallback frequency,
and recovery failure rates.
"""
import threading
from collections import defaultdict
from typing import Dict, Any


class DenseCriticalityDetector:
    """
    Identifies dense-critical layers based on semantic collapse sensitivity.
    A layer is dense-critical if it frequently exceeds the collapse threshold,
    or if repair fails to recover it.
    """
    COLLAPSE_DRIFT      = 0.12  # Drift level indicating collapse
    CRITICALITY_RATE    = 0.40  # Fraction of steps collapsing = critical

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.Lock()
        self._total_steps: Dict[int, int]   = defaultdict(int)
        self._collapse_steps: Dict[int, int] = defaultdict(int)
        self._repair_failed: Dict[int, int]  = defaultdict(int)

    def record_step(self, layer_idx: int, drift: float, repair_attempted: bool, repair_effective: bool):
        with self._lock:
            self._total_steps[layer_idx] += 1
            if drift >= self.COLLAPSE_DRIFT:
                self._collapse_steps[layer_idx] += 1
            if repair_attempted and not repair_effective:
                self._repair_failed[layer_idx] += 1

    def is_dense_critical(self, layer_idx: int) -> bool:
        with self._lock:
            total = self._total_steps.get(layer_idx, 0)
            if total == 0:
                return False
            collapse_rate = self._collapse_steps.get(layer_idx, 0) / total
            return collapse_rate >= self.CRITICALITY_RATE

    def get_criticality_map(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            result = {}
            for layer_idx in range(self.num_layers):
                total = self._total_steps.get(layer_idx, 0)
                if total == 0:
                    continue
                collapses = self._collapse_steps.get(layer_idx, 0)
                failed = self._repair_failed.get(layer_idx, 0)
                result[layer_idx] = {
                    "total_steps": total,
                    "collapse_rate": round(collapses / total, 4),
                    "repair_failure_count": failed,
                    "is_dense_critical": (collapses / total) >= self.CRITICALITY_RATE
                }
            return result

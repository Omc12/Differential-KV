"""
STAGE 2 - HSZ: Repair Effectiveness Analyzer
Phase 39.3 - Hybrid Semantic Zoning

Distinguishes repair PARTICIPATION from repair EFFECTIVENESS.
A repair only counts as effective if post-repair drift is materially lower
than pre-repair drift.
"""
import threading
from typing import Dict, Any


class RepairEffectivenessAnalyzer:
    """
    Tracks whether semantic repair activations actually reduced KL divergence.
    """
    EFFECTIVENESS_DELTA = 0.02  # Minimum required drift reduction to count as effective

    def __init__(self):
        self._lock = threading.Lock()
        self._activations = 0
        self._effective = 0
        self._ineffective = 0
        self._total_drift_reduction = 0.0
        # layer-level stats
        self._layer_activations: Dict[int, int] = {}
        self._layer_effective: Dict[int, int] = {}
        self._layer_drift_delta: Dict[int, float] = {}

    def record_repair_attempt(self, layer_idx: int, drift_before: float, drift_after: float):
        """
        Called once per repair event with the KL divergence before and after.
        """
        delta = drift_before - drift_after  # positive = improvement
        with self._lock:
            self._activations += 1
            self._layer_activations[layer_idx] = self._layer_activations.get(layer_idx, 0) + 1
            self._layer_drift_delta[layer_idx] = (
                self._layer_drift_delta.get(layer_idx, 0.0) + delta
            )

            if delta >= self.EFFECTIVENESS_DELTA:
                self._effective += 1
                self._total_drift_reduction += delta
                self._layer_effective[layer_idx] = self._layer_effective.get(layer_idx, 0) + 1
            else:
                self._ineffective += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._activations, 1)
            return {
                "total_activations": self._activations,
                "effective_repairs": self._effective,
                "ineffective_repairs": self._ineffective,
                "effectiveness_rate": round(self._effective / total, 4),
                "avg_drift_reduction": round(
                    self._total_drift_reduction / max(self._effective, 1), 6
                ),
            }

    def get_layer_effectiveness(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            result = {}
            for layer_idx, acts in self._layer_activations.items():
                eff = self._layer_effective.get(layer_idx, 0)
                delta = self._layer_drift_delta.get(layer_idx, 0.0)
                result[layer_idx] = {
                    "activations": acts,
                    "effective": eff,
                    "rate": round(eff / max(acts, 1), 4),
                    "total_drift_reduction": round(delta, 6)
                }
            return result

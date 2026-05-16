"""
STAGE 2.5 - SRC: Long-Horizon Degradation Curve System
Phase 40.0 - Scientific Research Consolidation

Measures continuous semantic degradation over extended reasoning horizons.
"""
import threading
from typing import Dict, Any

class LongHorizonDegradationCurveSystem:
    def __init__(self):
        self._lock = threading.RLock()
        self._degradation_curve = []
        self._cumulative_drift = 0.0

    def record_step(self, step: int, layer_drift: float):
        with self._lock:
            self._cumulative_drift += layer_drift
            slope = self._cumulative_drift / max(step, 1)
            self._degradation_curve.append({"step": step, "cumulative_drift": self._cumulative_drift, "slope": slope})

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            slope = self._degradation_curve[-1]["slope"] if self._degradation_curve else 0.0
            return {
                "degradation_slope": round(slope, 4),
                "cumulative_drift": round(self._cumulative_drift, 4)
            }

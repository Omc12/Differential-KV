"""
STAGE 2 - OSE Hardening: Internal vs External Divergence Detector
Phase 39.7 - Objective Semantic Evaluation

Detects situations where internal equilibrium rises BUT external reasoning fidelity falls.
"""
import threading
from typing import Dict, Any

class InternalExternalDivergenceDetector:
    def __init__(self):
        self._lock = threading.RLock()
        self._divergence_events = 0
        self._total_checks = 0
        self._max_divergence_gap = 0.0

    def check_divergence(self, internal_equilibrium: float, external_fidelity: float):
        with self._lock:
            self._total_checks += 1
            # Divergence happens if equilibrium is high (>0.8) but fidelity is low (<0.6)
            gap = internal_equilibrium - external_fidelity
            if gap > 0.3:
                self._divergence_events += 1
            
            if gap > self._max_divergence_gap:
                self._max_divergence_gap = gap

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_checks, 1)
            return {
                "divergence_event_rate": round(self._divergence_events / total, 4),
                "max_divergence_gap": round(self._max_divergence_gap, 4)
            }

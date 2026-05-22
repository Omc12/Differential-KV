"""
STAGE 2 - OSE: Semantic Divergence Comparator
Phase 39.7 - Objective Semantic Evaluation

Measures TRUE semantic divergence between sparse and dense execution.
"""
import threading
from typing import Dict, Any

class SemanticDivergenceComparator:
    def __init__(self):
        self._lock = threading.RLock()
        self._divergence_events = 0
        self._total_checks = 0
        self._max_divergence = 0.0

    def record_divergence(self, step: int, kl_divergence: float):
        with self._lock:
            self._total_checks += 1
            if kl_divergence > 1.5:  # High divergence threshold
                self._divergence_events += 1
            if kl_divergence > self._max_divergence:
                self._max_divergence = kl_divergence

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_checks, 1)
            return {
                "divergence_event_rate": round(self._divergence_events / total, 4),
                "max_recorded_divergence": round(self._max_divergence, 4)
            }

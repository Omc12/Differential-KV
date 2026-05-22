"""
STAGE 2 - ARS: Reasoning Collapse Detector
Phase 39.8 - Adversarial Reasoning Stability

Detects true reasoning breakdown events under adversarial load.
Evaluates structure, not fluency.
"""
import threading
from typing import Dict, Any

class ReasoningCollapseDetector:
    def __init__(self):
        self._lock = threading.RLock()
        self._collapse_events = 0
        self._total_checks = 0

    def check_collapse(self, kl_divergence: float, reasoning_agreement: float, internal_drift: float):
        with self._lock:
            self._total_checks += 1
            # A structural reasoning collapse is defined by high internal drift combined with 
            # failed external reasoning agreement, independent of KL-divergence magnitude.
            if internal_drift > 1.5 and reasoning_agreement < 0.5:
                self._collapse_events += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_checks, 1)
            return {
                "reasoning_collapse_events": self._collapse_events,
                "reasoning_collapse_rate": round(self._collapse_events / total, 4)
            }

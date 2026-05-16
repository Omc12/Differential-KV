"""
STAGE 2 - OSE Hardening: Telemetry Overfitting Detector
Phase 39.7 - Objective Semantic Evaluation

Detects whether learned governance policies are optimizing telemetry metrics
instead of preserving true reasoning fidelity.
"""
import threading
from typing import Dict, Any

class TelemetryOverfittingDetector:
    def __init__(self):
        self._lock = threading.RLock()
        self._overfitting_events = 0
        self._total_checks = 0

    def check_overfitting(self, policy_confidence: float, eq_score: float, reasoning_agreement: float):
        with self._lock:
            self._total_checks += 1
            
            # Overfitting happens if the policy is highly confident and equilibrium is perfect,
            # BUT the actual reasoning agreement with dense reference is poor.
            if policy_confidence > 0.8 and eq_score > 0.9 and reasoning_agreement < 0.6:
                self._overfitting_events += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_checks, 1)
            return {
                "telemetry_overfitting_rate": round(self._overfitting_events / total, 4)
            }

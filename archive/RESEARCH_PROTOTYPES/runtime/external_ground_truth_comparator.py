"""
STAGE 2 - OSE Hardening: External Ground Truth Comparator
Phase 39.7 - Objective Semantic Evaluation

Forces major semantic claims to be validated against true dense-reference outputs.
External to the adaptive policy learner.
"""
import threading
from typing import Dict, Any

class ExternalGroundTruthComparator:
    def __init__(self):
        self._lock = threading.RLock()
        self._total_comparisons = 0
        self._ground_truth_agreements = 0

    def validate_ground_truth(self, sparse_correct: bool, dense_correct: bool):
        with self._lock:
            self._total_comparisons += 1
            if sparse_correct == dense_correct:
                self._ground_truth_agreements += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_comparisons, 1)
            return {
                "ground_truth_agreement_rate": round(self._ground_truth_agreements / total, 4)
            }

"""
STAGE 2 - ARS: Contradiction Persistence Analyzer
Phase 39.8 - Adversarial Reasoning Stability

Measures whether sparse governance introduces hidden contradictions over long chains.
"""
import threading
from typing import Dict, Any

class ContradictionPersistenceAnalyzer:
    def __init__(self):
        self._lock = threading.RLock()
        self._total_tests = 0
        self._introduced_contradictions = 0

    def evaluate_contradiction(self, sparse_correct: bool, dense_correct: bool, task_type: str):
        with self._lock:
            if task_type == "contradiction":
                self._total_tests += 1
                # If dense avoided it but sparse fell for it, a contradiction was introduced
                if dense_correct and not sparse_correct:
                    self._introduced_contradictions += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_tests, 1)
            return {
                "contradiction_emergence_rate": round(self._introduced_contradictions / total, 4)
            }

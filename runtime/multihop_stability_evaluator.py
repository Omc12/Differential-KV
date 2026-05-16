"""
STAGE 2 - ARS: Multi-Hop Stability Evaluator
Phase 39.8 - Adversarial Reasoning Stability

Measures whether sparse execution preserves multi-stage reasoning continuity.
"""
import threading
from typing import Dict, Any

class MultihopStabilityEvaluator:
    def __init__(self):
        self._lock = threading.RLock()
        self._hop_tests = 0
        self._successful_multihop_chains = 0

    def evaluate_multihop(self, is_sparse_correct: bool, is_dense_correct: bool, hop_length: int):
        with self._lock:
            self._hop_tests += 1
            if is_sparse_correct or not is_dense_correct:
                self._successful_multihop_chains += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._hop_tests, 1)
            return {
                "multihop_stability_rate": round(self._successful_multihop_chains / total, 4)
            }

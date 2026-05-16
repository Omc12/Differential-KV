"""
STAGE 2 - OSE: Long-Context Recall Evaluator
Phase 39.7 - Objective Semantic Evaluation

Measures whether sparse execution preserves long-range contextual memory.
"""
import threading
from typing import Dict, Any

class LongContextRecallEvaluator:
    def __init__(self):
        self._lock = threading.RLock()
        self._recall_tests = 0
        self._successful_recalls = 0

    def evaluate_recall(self, is_dense_correct: bool, is_sparse_correct: bool):
        with self._lock:
            self._recall_tests += 1
            if is_sparse_correct or not is_dense_correct:
                # If sparse is correct, or dense also failed (meaning it's not sparsity's fault)
                self._successful_recalls += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._recall_tests, 1)
            return {
                "long_context_recall_fidelity": round(self._successful_recalls / total, 4)
            }

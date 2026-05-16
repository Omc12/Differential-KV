"""
STAGE 2 - OSE: Reasoning Integrity Benchmark Harness
Phase 39.7 - Objective Semantic Evaluation

Runs objective reasoning tasks through dense baseline and sparse-governed runtime
to compare correctness and consistency.
"""
import threading
from typing import Dict, Any, List

class ReasoningIntegrityBenchmarkHarness:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks_run = 0
        self._correct_dense = 0
        self._correct_sparse = 0
        self._reasoning_agreement = 0
        
        # Hardcoded simulated small tasks for runtime evaluation
        self.tasks = [
            {"type": "logic", "prompt": "If A is B and B is C, is A C?", "expected": "yes"},
            {"type": "math", "prompt": "What is 15 * 4?", "expected": "60"},
            {"type": "fact", "prompt": "What is the capital of France?", "expected": "paris"},
        ]

    def evaluate_task(self, dense_output: str, sparse_output: str, expected: str):
        with self._lock:
            self._tasks_run += 1
            
            expected_lower = expected.lower()
            d_out = dense_output.lower()
            s_out = sparse_output.lower()
            
            d_correct = expected_lower in d_out
            s_correct = expected_lower in s_out
            
            if d_correct: self._correct_dense += 1
            if s_correct: self._correct_sparse += 1
            
            if d_correct == s_correct:
                self._reasoning_agreement += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._tasks_run, 1)
            return {
                "dense_accuracy": round(self._correct_dense / total, 4),
                "sparse_accuracy": round(self._correct_sparse / total, 4),
                "reasoning_agreement_rate": round(self._reasoning_agreement / total, 4)
            }

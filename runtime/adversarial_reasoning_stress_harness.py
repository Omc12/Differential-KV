"""
STAGE 2 - ARS: Adversarial Reasoning Stress Harness
Phase 39.8 - Adversarial Reasoning Stability

Runs high-fragility reasoning tasks (contradictions, multi-hop, delayed logic)
through dense baseline and sparse-governed runtime.
"""
import threading
from typing import Dict, Any, List

class AdversarialReasoningStressHarness:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks_run = 0
        self._correct_dense = 0
        self._correct_sparse = 0
        self._reasoning_agreement = 0
        
        # Simulated adversarial tasks
        self.tasks = [
            {"type": "contradiction", "prompt": "If A=1 and A=2, what is A?", "expected": "contradiction"},
            {"type": "multi-hop", "prompt": "X is in Y. Y is in Z. Where is X?", "expected": "Z"},
            {"type": "delayed_dependency", "prompt": "Remember pin 1234. [...] What was the pin?", "expected": "1234"},
            {"type": "long_arithmetic", "prompt": "(((1+2)*3)-4)/5 = ?", "expected": "1"},
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
                "adversarial_dense_acc": round(self._correct_dense / total, 4),
                "adversarial_sparse_acc": round(self._correct_sparse / total, 4),
                "adversarial_agreement_rate": round(self._reasoning_agreement / total, 4)
            }

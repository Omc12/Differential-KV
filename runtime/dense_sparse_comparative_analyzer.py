"""
STAGE 2 - RBT: Dense vs Sparse Comparative Analyzer
Phase 39.9 - Rigorous Benchmark Triangulation

Directly compares reasoning trajectories between dense and sparse paths.
"""
import threading
from typing import Dict, Any

class DenseSparseComparativeAnalyzer:
    def __init__(self):
        self._lock = threading.RLock()
        self._total_comparisons = 0
        self._agreements = 0

    def analyze(self, sparse_correct: bool, dense_correct: bool):
        with self._lock:
            self._total_comparisons += 1
            if sparse_correct == dense_correct:
                self._agreements += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_comparisons, 1)
            return {
                "dense_sparse_agreement_rate": round(self._agreements / total, 4)
            }

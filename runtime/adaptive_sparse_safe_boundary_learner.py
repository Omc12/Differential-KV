"""
STAGE 2 - ASI: Adaptive Sparse-Safe Boundary Learner
Phase 39.6 - Adaptive Semantic Intelligence

Learns where the semantic safety boundary actually lies without hardcoded assumptions.
Gradually improves at estimating safe sparse ratios and continuity lengths.
"""
import threading
from typing import Dict, Any, List

class AdaptiveSparseSafeBoundaryLearner:
    def __init__(self):
        self._lock = threading.RLock()
        
        # Start with conservative estimates, learn upwards or downwards
        self._safe_chain_length = 10.0
        self._safe_sparse_ratio = 0.5
        
        self._learning_rate = 0.05

    def record_chain_outcome(self, chain_length: int, collapsed: bool):
        with self._lock:
            if collapsed:
                # Chain was too long, boundary is lower
                if chain_length <= self._safe_chain_length:
                    self._safe_chain_length *= (1.0 - self._learning_rate)
            else:
                # Chain survived, boundary might be higher
                if chain_length >= self._safe_chain_length:
                    self._safe_chain_length *= (1.0 + self._learning_rate * 0.5) # Slower growth

    def record_ratio_outcome(self, current_ratio: float, semantic_equilibrium: float):
        with self._lock:
            if semantic_equilibrium < 0.6:
                # Too sparse, equilibrium failing
                if current_ratio >= self._safe_sparse_ratio:
                    self._safe_sparse_ratio = max(0.1, self._safe_sparse_ratio - self._learning_rate)
            elif semantic_equilibrium > 0.9:
                # Very stable, maybe can push sparsity
                self._safe_sparse_ratio = min(0.95, self._safe_sparse_ratio + self._learning_rate * 0.2)

    def get_boundaries(self) -> Dict[str, float]:
        with self._lock:
            return {
                "safe_chain_length": round(self._safe_chain_length, 2),
                "safe_sparse_ratio": round(self._safe_sparse_ratio, 4)
            }

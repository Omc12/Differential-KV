"""
STAGE 2 - RBT: Failure Boundary Mapper
Phase 39.9 - Rigorous Benchmark Triangulation

Identifies exactly WHERE semantic stability begins to fail.
"""
import threading
from typing import Dict, Any

class FailureBoundaryMapper:
    def __init__(self):
        self._lock = threading.RLock()
        # Mapping limits
        self._max_safe_sparse_ratio = 1.0
        self._max_safe_context_len = 8000
        self._max_safe_dependency_depth = 50

    def record_failure(self, sparse_ratio: float, context_len: int, dependency_depth: int):
        with self._lock:
            # If it failed, the boundary is slightly below where it failed
            if sparse_ratio <= self._max_safe_sparse_ratio:
                self._max_safe_sparse_ratio = sparse_ratio * 0.95
            
            if context_len < self._max_safe_context_len:
                self._max_safe_context_len = int(context_len * 0.95)
                
            if dependency_depth < self._max_safe_dependency_depth:
                self._max_safe_dependency_depth = max(1, int(dependency_depth * 0.9))

    def get_boundaries(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "limit_sparse_ratio": round(self._max_safe_sparse_ratio, 4),
                "limit_context_len": self._max_safe_context_len,
                "limit_dependency_depth": self._max_safe_dependency_depth
            }

"""
PCR Phase 41.4.5: Dense vs Sparse Physical Comparator.
Purpose: Compare actual dense vs sparse execution (real latency delta, VRAM delta).
"""

from typing import Dict, Any

class DenseSparsePhysicalComparator:
    def __init__(self):
        self._dense_kernel_launches = 0
        self._sparse_kernel_launches = 0
        self._dense_latency_sum = 0.0
        self._sparse_latency_sum = 0.0

    def record_dense_pass(self, latency_ms: float, kernels: int):
        self._dense_kernel_launches += kernels
        self._dense_latency_sum += latency_ms

    def record_sparse_pass(self, latency_ms: float, kernels: int):
        self._sparse_kernel_launches += kernels
        self._sparse_latency_sum += latency_ms

    def get_stats(self) -> Dict[str, Any]:
        delta_pct = 0.0
        if self._dense_latency_sum > 0:
            delta_pct = ((self._dense_latency_sum - self._sparse_latency_sum) / self._dense_latency_sum) * 100.0
            
        return {
            "dense_kernel_launches": self._dense_kernel_launches,
            "sparse_kernel_launches": self._sparse_kernel_launches,
            "dense_latency_ms": self._dense_latency_sum,
            "sparse_latency_ms": self._sparse_latency_sum,
            "sparse_vs_dense_compute_delta_pct": delta_pct
        }

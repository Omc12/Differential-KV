"""
SIP Phase 41.2: Sparse Participation Reality Meter.

Purpose: Measure REAL sparse execution participation (no inferred metrics).
"""
from typing import Dict, Any

class SparseParticipationRealityMeter:
    def __init__(self):
        self._sparse_routed_tokens = 0
        self._dense_routed_tokens = 0
        self._repair_routed_tokens = 0
        self._sparse_governance_decisions = 0
        self._sparse_metadata_hits = 0
        self._active_sparse_kv_usage = 0

    def record_sparse_token(self, count: int = 1):
        self._sparse_routed_tokens += count

    def record_dense_token(self, count: int = 1):
        self._dense_routed_tokens += count

    def record_repair_token(self, count: int = 1):
        self._repair_routed_tokens += count

    def record_governance_decision(self, count: int = 1):
        self._sparse_governance_decisions += count
        
    def record_metadata_hit(self, count: int = 1):
        self._sparse_metadata_hits += count
        
    def record_active_kv_usage(self, count: int = 1):
        self._active_sparse_kv_usage += count

    def get_participation_stats(self) -> Dict[str, Any]:
        total_tokens = (
            self._sparse_routed_tokens + 
            self._dense_routed_tokens + 
            self._repair_routed_tokens
        )
        
        return {
            "sparse_routed_tokens": self._sparse_routed_tokens,
            "dense_routed_tokens": self._dense_routed_tokens,
            "repair_routed_tokens": self._repair_routed_tokens,
            "sparse_governance_decisions": self._sparse_governance_decisions,
            "sparse_metadata_hits": self._sparse_metadata_hits,
            "active_sparse_kv_usage": self._active_sparse_kv_usage,
            "sparse_participation_ratio": self._sparse_routed_tokens / total_tokens if total_tokens > 0 else 0.0,
            "repair_ratio": self._repair_routed_tokens / total_tokens if total_tokens > 0 else 0.0,
            "dense_ratio": self._dense_routed_tokens / total_tokens if total_tokens > 0 else 0.0,
            "has_material_participation": self._sparse_routed_tokens > 0
        }

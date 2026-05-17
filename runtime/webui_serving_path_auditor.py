"""
SIP Phase 41.2: WebUI Serving Path Auditor.

Purpose: Audit the REAL browser serving path to ensure it uses the Differential KV runtime,
not a dense bypass or a generic wrapper.
"""
from typing import Dict, Any

class WebUIServingPathAuditor:
    def __init__(self):
        self._path_events = []
        self._bypass_events = 0
        self._dense_only_compatibility = 0
        self._simplified_wrapper_used = 0
        self._disabled_sparse_path = 0
        self._disabled_native_path = 0

    def record_path_event(self, request_id: str, path_segment: str, is_bypass: bool = False):
        self._path_events.append({
            "request_id": request_id,
            "segment": path_segment,
            "is_bypass": is_bypass
        })
        if is_bypass:
            self._bypass_events += 1

    def flag_dense_only_compatibility(self, request_id: str):
        self._dense_only_compatibility += 1
        self.record_path_event(request_id, "DenseOnlyCompatibilityBypass", is_bypass=True)

    def flag_simplified_wrapper(self, request_id: str):
        self._simplified_wrapper_used += 1
        self.record_path_event(request_id, "SimplifiedOpenAIWrapperBypass", is_bypass=True)

    def flag_disabled_sparse(self, request_id: str):
        self._disabled_sparse_path += 1
        self.record_path_event(request_id, "DisabledSparseGovernanceBypass", is_bypass=True)
        
    def flag_disabled_native(self, request_id: str):
        self._disabled_native_path += 1
        self.record_path_event(request_id, "DisabledNativeAccelerationBypass", is_bypass=True)

    def get_audit_stats(self) -> Dict[str, Any]:
        total_events = len(self._path_events)
        return {
            "total_path_events": total_events,
            "total_bypasses": self._bypass_events,
            "dense_compatibility_flags": self._dense_only_compatibility,
            "simplified_wrapper_flags": self._simplified_wrapper_used,
            "disabled_sparse_flags": self._disabled_sparse_path,
            "disabled_native_flags": self._disabled_native_path,
            "path_integrity_score": (total_events - self._bypass_events) / total_events if total_events > 0 else 1.0,
            "is_clean": self._bypass_events == 0
        }

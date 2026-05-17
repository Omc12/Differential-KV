"""
SIP Phase 41.2: Native Path Activation Verifier.

Purpose: Prove native modules are ACTUALLY executing.
"""
from typing import Dict, Any

class NativePathActivationVerifier:
    def __init__(self):
        self._native_scheduler_calls = 0
        self._native_sparse_metadata_lookups = 0
        self._native_telemetry_increments = 0
        self._fallback_path_calls = 0

    def record_native_scheduler_call(self, count: int = 1):
        self._native_scheduler_calls += count

    def record_native_metadata_lookup(self, count: int = 1):
        self._native_sparse_metadata_lookups += count

    def record_native_telemetry_increment(self, count: int = 1):
        self._native_telemetry_increments += count
        
    def record_fallback_path_call(self, count: int = 1):
        self._fallback_path_calls += count

    def get_activation_stats(self) -> Dict[str, Any]:
        total_native = (
            self._native_scheduler_calls + 
            self._native_sparse_metadata_lookups + 
            self._native_telemetry_increments
        )
        total_calls = total_native + self._fallback_path_calls
        
        return {
            "native_scheduler_calls": self._native_scheduler_calls,
            "native_metadata_lookups": self._native_sparse_metadata_lookups,
            "native_telemetry_increments": self._native_telemetry_increments,
            "fallback_path_calls": self._fallback_path_calls,
            "total_native_calls": total_native,
            "native_execution_ratio": total_native / total_calls if total_calls > 0 else 0.0,
            "is_native_active": total_native > 0 and self._fallback_path_calls == 0
        }

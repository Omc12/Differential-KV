"""
cbp_integrity_guard.py

Final validation guard for CBP benchmarks.
Fails if telemetry is inconsistent, overhead is hidden, or comparisons are unfair.
"""

from typing import Dict, Any, List
import logging

class CBPIntegrityGuard:
    """
    Enforces strict integrity rules on CBP benchmark results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("CBPIntegrityGuard")

    def validate_final_results(self, metrics: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
        """
        Validates the entire benchmark result set against CBP standards.
        """
        violations = []
        
        # 1. Telemetry Consistency
        if metrics.get("sustained_tps", 0) <= 0:
            violations.append("Sustained TPS must be positive.")
            
        # 2. Serving Overhead
        if not manifest.get("serving_overhead_included", False):
            violations.append("Serving overhead must be included in production benchmarks.")
            
        # 3. Sparse Participation
        if metrics.get("sparse_runtime_pct", 0) < 50.0:
            violations.append(f"Sparse runtime participation too low: {metrics.get('sparse_runtime_pct', 0):.1f}%")
            
        # 4. Physical Plausibility Check
        # 9000+ TPS is not physically plausible for real end-to-end autoregressive decode
        if metrics.get("sustained_tps", 0) > 1000.0:
            violations.append(f"UNPLAUSIBLE TPS detected ({metrics.get('sustained_tps', 0):.2f}). Subsystem contamination suspected.")

        # 5. Synthetic Metric Detection
        if manifest.get("telemetry_scope", {}).get("synthetic_accounting", False):
            violations.append("Synthetic accounting detected in production benchmark.")
            
        # 6. Scope Validation
        scope = manifest.get("telemetry_scope", {})
        if not scope.get("wall_clock", False):
            violations.append("Production benchmark MUST use real wall-clock timing.")
        if not scope.get("gpu_allocations", False):
            violations.append("Production benchmark MUST use real GPU memory residency.")

        if violations:
            for v in violations:
                self.logger.error(f"[CBP GUARD] VIOLATION: {v}")
            return False
            
        self.logger.info("[CBP GUARD] All integrity checks passed. Metrics are physically plausible.")
        return True

# Global instance
cbp_integrity_guard = CBPIntegrityGuard()

import logging
from typing import Dict, Any, List

class EOMIntegrityGuard:
    """
    EOM MODULE 6: Prevents synthetic metrics from contaminating EOM results.
    Ensures that EOM gains are derived from real serving optimizations.
    """
    def __init__(self):
        self.logger = logging.getLogger("EOMIntegrityGuard")

    def validate_eom_results(self, metrics: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
        violations = []
        
        # 1. Synthetic Batching Check
        if manifest.get("synthetic_batching", False):
            violations.append("Synthetic batching detected. Must use real scheduler.")
            
        # 2. Serving Overhead Inclusion
        if metrics.get("serving_overhead_ratio", 0) < 0.05:
            # If serving overhead is suspiciously low, it might be bypassed
            violations.append(f"Suspiciously low serving overhead ({metrics.get('serving_overhead_ratio', 0)*100:.1f}%). Audit required.")
            
        # 3. Occupancy Metrics Check
        if manifest.get("occupancy_metrics", {}).get("synthetic", False):
            violations.append("Synthetic occupancy metrics detected.")
            
        # 4. Sparse Runtime Connection
        if not manifest.get("sparse_runtime_attached", False):
            violations.append("Sparse runtime detached from serving layer.")

        if violations:
            for v in violations:
                self.logger.error(f"[EOM GUARD] VIOLATION: {v}")
            return False
            
        self.logger.info("[EOM GUARD] EOM results validated. Optimizations are physically grounded.")
        return True

# Global instance
eom_integrity_guard = EOMIntegrityGuard()

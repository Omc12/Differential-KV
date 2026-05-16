import logging
from typing import Dict, Any

class PDMIntegrityGuard:
    """
    Validation MUST FAIL if:
    - deployment not reproducible
    - serving silently degrades
    - crashes corrupt sessions
    - telemetry missing
    - sparse runtime detached
    """
    def __init__(self):
        self.logger = logging.getLogger("PDMIntegrityGuard")

    def validate_pdm_results(self, metrics: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
        self.logger.info("Starting PDM Integrity Audit...")
        
        # 1. Reproducibility Check
        if not metrics.get("deployment_reproducible", False):
            self.logger.error("PDM Integrity FAILED: Deployment environment not reproducible.")
            return False
            
        # 2. Recovery Check
        recovery_rate = metrics.get("recovery_success_rate", 0)
        if recovery_rate < 1.0: # Expect perfect recovery for this pass
            self.logger.error(f"PDM Integrity FAILED: Recovery success rate ({recovery_rate*100:.1f}%) below 100%.")
            return False
            
        # 3. Telemetry Persistence
        if not metrics.get("telemetry_persisted", False):
            self.logger.error("PDM Integrity FAILED: Telemetry did not survive runtime restart.")
            return False
            
        # 4. Sparse Participation Retention
        min_sparse = manifest.get("min_sparse_ratio", 0.95)
        avg_sparse = metrics.get("avg_sparse_ratio", 0)
        if avg_sparse < min_sparse:
            self.logger.error(f"PDM Integrity FAILED: Sparse participation ({avg_sparse:.2f}) degraded under pressure.")
            return False
            
        # 5. Stability Index
        min_stability = manifest.get("min_stability_index", 90.0)
        stability_idx = metrics.get("operational_stability_index", 0)
        if stability_idx < min_stability:
            self.logger.error(f"PDM Integrity FAILED: Operational Stability Index ({stability_idx:.2f}) too low.")
            return False

        self.logger.info("PDM Integrity Audit PASSED.")
        return True

pdm_integrity_guard = PDMIntegrityGuard()

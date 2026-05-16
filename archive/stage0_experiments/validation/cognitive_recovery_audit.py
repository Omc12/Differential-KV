from validation.state_integrity_auditor import StateIntegrityAuditor
from validation.recursive_stability_audit import RecursiveStabilityAudit
from validation.policy_regression_guard import PolicyRegressionGuard
from typing import Dict, Any, List

class CognitiveRecoveryAudit:
    """
    General audit for recovered systems.
    Combines integrity, stability, and regression checks.
    """
    def __init__(self):
        self.integrity_auditor = StateIntegrityAuditor()
        self.stability_auditor = RecursiveStabilityAudit()
        self.regression_guard = PolicyRegressionGuard()

    def perform_full_audit(self, state: Dict[str, Any], trace: List[Dict[str, float]], current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Performs a comprehensive audit of all CCR components."""
        integrity = self.integrity_auditor.audit_state(state)
        stability = self.stability_auditor.audit_trace(trace)
        regression = self.regression_guard.check_regression(current_metrics)
        
        overall_pass = integrity and (stability["status"] == "PASS") and (regression["status"] == "PASS")
        
        return {
            "overall_status": "PASS" if overall_pass else "FAIL",
            "integrity": self.integrity_auditor.get_audit_report(),
            "stability": stability,
            "regression": regression
        }

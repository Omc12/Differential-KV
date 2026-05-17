import time
from typing import Dict, Any, List

class ExecutionPathCorrelationAuditor:
    """
    2. Execution Path Correlation Auditor
    
    Tracks real inference path traversal, verifies actual execution path,
    detects bypassed layers, and correlates emitted generation with execution flow.
    """
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.audits = []
        self.total_traversed_layers = 0
        self.total_expected_layers = 0

    def audit_step(self, step: int, expected_path: List[int], actual_path: List[int]) -> Dict[str, Any]:
        """
        Audit a step's execution path, comparing the actual layer traversal to expected layers.
        """
        expected_set = set(expected_path)
        actual_set = set(actual_path)
        
        overlap = expected_set.intersection(actual_set)
        consistency = len(overlap) / max(len(expected_set), 1)
        
        self.total_traversed_layers += len(actual_set)
        self.total_expected_layers += len(expected_set)
        
        record = {
            "step": step,
            "timestamp": time.time(),
            "expected_layers": expected_path,
            "actual_layers": actual_path,
            "consistency_ratio": consistency,
            "bypassed_layers": list(expected_set - actual_set)
        }
        self.audits.append(record)
        return record

    def get_participation_ratio(self) -> float:
        """
        Returns the overall ratio of actual layer traversals to planned/expected layer traversals.
        """
        if not self.audits:
            return 100.0
        
        # We calculate the participation ratio as average actual to expected
        ratios = [len(a["actual_layers"]) / max(len(a["expected_layers"]), 1) for a in self.audits]
        return (sum(ratios) / len(ratios)) * 100.0

    def get_path_consistency(self) -> float:
        """
        Returns execution path consistency based on how closely actual path matches expected path.
        """
        if not self.audits:
            return 100.0
        consistencies = [a["consistency_ratio"] for a in self.audits]
        return (sum(consistencies) / len(consistencies)) * 100.0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_audited_steps": len(self.audits),
            "runtime_participation_percent": self.get_participation_ratio(),
            "execution_path_consistency_percent": self.get_path_consistency(),
            "status": "AUTHENTIC" if self.get_path_consistency() >= 99.0 else "DRIFTED"
        }

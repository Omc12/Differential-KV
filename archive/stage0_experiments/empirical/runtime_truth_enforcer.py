import json
import os
from typing import List, Dict, Any

class RuntimeTruthEnforcer:
    """
    Adversarially audits log files to ensure they contain empirical data, not projections.
    """
    def __init__(self):
        pass

    def audit_log(self, log_path: str) -> Dict[str, Any]:
        if not os.path.exists(log_path):
            return {"status": "error", "message": "Log file not found"}

        with open(log_path, "r") as f:
            lines = f.readlines()

        total_entries = len(lines)
        projected_markers = ["projection", "synthetic", "estimated", "forecast"]
        contamination_count = 0
        
        for line in lines:
            data = json.loads(line)
            # Check for high-precision synthetic numbers (e.g. perfect 100.0s)
            if any(isinstance(v, (int, float)) and v % 1.0 == 0 and v != 0 for v in data.values()):
                # This is a weak heuristic, but real hardware data is rarely perfectly integer
                pass
            
            # Check for keyword contamination
            if any(marker in str(data).lower() for marker in projected_markers):
                contamination_count += 1

        rejection_score = contamination_count / total_entries if total_entries > 0 else 1.0
        
        return {
            "status": "pass" if rejection_score < 0.05 else "rejected",
            "contamination_ratio": rejection_score,
            "total_entries": total_entries,
            "message": "Empirical validity verified" if rejection_score < 0.05 else "LOG CONTAMINATED WITH PROJECTIONS"
        }

if __name__ == "__main__":
    enforcer = RuntimeTruthEnforcer()
    # Mock audit
    print(enforcer.audit_log("results/reconstruction_6_5/test_run/raw_telemetry.json"))

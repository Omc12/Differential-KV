import os
from typing import Dict, Any

class PSRIntegrityGuard:
    """
    PSR System 6: PSR Integrity Guard.
    Ensures that validation fails if serving realism is bypassed.
    """
    def __init__(self):
        self.violations = []

    def validate_psr_config(self, config: Dict[str, Any]):
        if not config.get("streaming_enabled", False):
            self.violations.append("Streaming disabled: PSR requires end-to-end streaming realism.")
        
        if not config.get("concurrency_levels", []):
            self.violations.append("Concurrency levels missing: PSR requires multi-user load testing.")
        elif max(config.get("concurrency_levels", [0])) < 1:
            self.violations.append("Concurrency too low: PSR requires multi-user load.")
        
        if not config.get("include_tokenizer", False):
            self.violations.append("Tokenizer excluded: Serving overhead must include tokenization.")

    def audit_telemetry(self, telemetry_report: Dict[str, Any]):
        if telemetry_report.get("serving_overhead_ratio", 0) == 0:
            self.violations.append("Zero serving overhead detected: Isolated decode loop suspected.")
        
        if telemetry_report.get("p99_itl_ms", 0) == 0:
            self.violations.append("ITL metric missing: Streaming jitter not tracked.")

    def check(self) -> bool:
        if self.violations:
            print("!!! PSR INTEGRITY FAILURE !!!")
            for v in self.violations:
                print(f"  - {v}")
            return False
        print("PSR Integrity Guard: PASSED")
        return True

    def enforce_serving_overhead(self, duration_ms: float):
        """Artificially ensures a minimum serving overhead if none is detected (for testing guard)."""
        if duration_ms < 0.1:
            # This should be caught by the audit
            pass

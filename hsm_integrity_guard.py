from typing import Dict, Any

class HSMIntegrityGuard:
    """
    HSM System 5: HSM Integrity Guard.
    Validates that serving is GPU-backed and materially active.
    """
    def __init__(self):
        self.violations = []

    def validate_hsm_state(self, residency_report: Dict[str, Any], telemetry_report: Dict[str, Any]):
        # 1. Model Residency Check
        if not residency_report.get("sustained_weight_residency", False):
            self.violations.append("Weight migration detected: Model weights not pinned to GPU.")
        
        # 2. VRAM Materiality Check
        # For a 7B model, we expect significant VRAM allocated
        if residency_report.get("vram_allocated_gb", 0) < 5.0: 
            self.violations.append(f"Mock residency suspected: VRAM allocated too low ({residency_report.get('vram_allocated_gb', 0):.2f} GB)")

        # 3. Concurrent Decode Reality Check
        if telemetry_report.get("system_tps", 0) == 0:
            self.violations.append("System TPS is zero: Concurrent decode engine inactive.")
            
        # 4. Heavy Concurrency Check
        if telemetry_report.get("total_requests", 0) < 10:
             self.violations.append("Insufficient request volume: Serving pressure synthetic.")

    def check(self) -> bool:
        if self.violations:
            print("!!! HSM INTEGRITY FAILURE !!!")
            for v in self.violations:
                print(f"  - {v}")
            return False
        print("HSM Integrity Guard: PASSED")
        return True

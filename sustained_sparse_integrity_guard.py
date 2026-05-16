from typing import Dict, Any, List

class SustainedSparseIntegrityGuard:
    """
    Detects fake materialization (e.g., idle GPU, dense fallback dominance).
    """
    def __init__(self):
        self.violations = []

    def validate_sustained_state(self, metrics: Dict[str, Any]):
        """
        Validates that sparse execution is sustained and hardware-visible.
        """
        # 1. Check for idle collapse
        if metrics.get("triton_kernel_runtime_percent", 0) < 10.0:
            self.violations.append("Triton kernels not dominating (runtime < 10%)")

        # 2. Check for dense fallback dominance
        if metrics.get("dense_fallback_count", 0) > 5:
            self.violations.append("Too many dense fallbacks during sustained decode")

        # 3. Check for occupancy stability
        if metrics.get("occupancy_stability_index", 0) < 0.5:
            self.violations.append("Occupancy unstable during execution")

    def check_integrity(self) -> bool:
        if self.violations:
            print("\n" + "!"*40)
            print("SHM INTEGRITY VIOLATIONS DETECTED")
            print("!"*40)
            for v in self.violations:
                print(f" - {v}")
            print("!"*40 + "\n")
            return False
        return True

guard = SustainedSparseIntegrityGuard()

from typing import Dict, Any, List

class SOCIntegrityGuard:
    """
    Detects fragmented sparse execution and unstable occupancy.
    Ensures that SOC achieves material consolidation.
    """
    def __init__(self):
        self.violations = []

    def validate_soc_state(self, metrics: Dict[str, Any]):
        """
        Validates that sparse execution is consolidated and efficient.
        """
        # 1. Check for launch fragmentation
        if metrics.get("launch_overhead_ratio", 1.0) > 0.5:
            self.violations.append("Sparse launches too fragmented (overhead > 50%)")

        # 2. Check for occupancy stability
        if metrics.get("occupancy_stability_index", 0) < 0.8:
            self.violations.append("Occupancy unstable during consolidated decode")

        # 3. Check for work window duration
        if metrics.get("triton_runtime_percent", 0) < 10.0:
             self.violations.append("Consolidated sparse runtime insufficient (<10%)")

    def check_integrity(self) -> bool:
        if self.violations:
            print("\n" + "!"*40)
            print("SOC INTEGRITY VIOLATIONS DETECTED")
            print("!"*40)
            for v in self.violations:
                print(f" - {v}")
            print("!"*40 + "\n")
            return False
        return True

guard = SOCIntegrityGuard()

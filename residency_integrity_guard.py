from typing import Dict, Any, List

class ResidencyIntegrityGuard:
    """
    Detects residency collapse and partial CPU execution.
    Ensures that Differential KV is materially GPU-resident.
    """
    def __init__(self):
        self.violations = []

    def validate_residency(self, metrics: Dict[str, Any]):
        """
        Validates that weights and compute are sustained on GPU.
        """
        # 1. Check for weight offloading
        if not metrics.get("sustained_weight_residency", True):
            self.violations.append("Model weights partially offloaded from GPU")

        # 2. Check for VRAM collapse
        if metrics.get("total_model_vram_gb", 0) < 0.1: # Threshold for opt-125m
             self.violations.append("Material VRAM residency missing (<0.1GB)")

        # 3. Check for detached runtime
        if not metrics.get("full_path_materialized", True):
            self.violations.append("Sparse runtime detached from full transformer path")

    def check_integrity(self) -> bool:
        if self.violations:
            print("\n" + "!"*40)
            print("FRM INTEGRITY VIOLATIONS DETECTED")
            print("!"*40)
            for v in self.violations:
                print(f" - {v}")
            print("!"*40 + "\n")
            return False
        return True

guard = ResidencyIntegrityGuard()

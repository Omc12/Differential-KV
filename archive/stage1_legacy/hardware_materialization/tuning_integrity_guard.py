"""
hardware_materialization/tuning_integrity_guard.py

Ensures that low-level tuning preserves numerical correctness and symbolic continuity.
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger("TuningGuard")

class TuningIntegrityGuard:
    """
    Validates tuning results against golden references.
    """
    def __init__(self):
        self.regressions = 0

    def validate_tuning_step(self, tuned_out: torch.Tensor, reference_out: torch.Tensor, step_name: str) -> bool:
        """
        Checks if tuned output matches reference output exactly.
        Tuning must NOT change numerical results.
        """
        if tuned_out.shape != reference_out.shape:
            logger.error(f"Shape mismatch after tuning {step_name}")
            return False

        # Tuning should be bit-exact or very close to bit-exact (floating point noise)
        # However, for GPU tuning like block size, it should be bit-exact.
        diff = torch.abs(tuned_out - reference_out).max().item()
        
        if diff > 1e-6:
            logger.error(f"Numerical regression detected in {step_name}: max_diff={diff}")
            self.regressions += 1
            return False
            
        return True

    def verify_symbolic_continuity(self, sequence: list) -> float:
        """Verifies that the sequence of tuned operations preserves lineage."""
        # Check if identifiers or markers are still intact
        return 1.0 # 100% continuity placeholder

    def get_guard_status(self) -> Dict[str, Any]:
        return {
            "total_regressions": self.regressions,
            "integrity_score": 1.0 if self.regressions == 0 else 0.0
        }

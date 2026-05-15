"""
hardware_materialization/hardware_materialization_integrity_guard.py

Validates numerical consistency between hardware-backed and fallback paths.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("IntegrityGuard")

class HardwareMaterializationIntegrityGuard:
    """
    Ensures that materialization doesn't break symbolic continuity or numerical stability.
    """
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.validation_history = []

    def validate_outputs(self, hardware_out: torch.Tensor, fallback_out: torch.Tensor, context: str = "op"):
        """
        Compares hardware output vs fallback output.
        Returns a consistency score (1.0 = perfect match).
        """
        if hardware_out.shape != fallback_out.shape:
            logger.error(f"Shape mismatch in {context}: {hardware_out.shape} vs {fallback_out.shape}")
            return 0.0

        # Numerical comparison
        diff = torch.abs(hardware_out - fallback_out)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Cosine similarity for direction check
        cos_sim = F.cosine_similarity(hardware_out.flatten(), fallback_out.flatten(), dim=0).item()
        
        status = "PASS" if max_diff < self.tolerance else "FAIL"
        
        metrics = {
            "context": context,
            "status": status,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "cosine_similarity": cos_sim
        }
        
        self.validation_history.append(metrics)
        
        if status == "FAIL":
            logger.warning(f"Integrity check failed for {context}: max_diff={max_diff:.6f}, sim={cos_sim:.6f}")
            
        return cos_sim if status == "PASS" else 0.0

    def get_summary(self):
        if not self.validation_history:
            return "No validation data."
            
        passes = sum(1 for v in self.validation_history if v["status"] == "PASS")
        total = len(self.validation_history)
        return f"Integrity Guard: {passes}/{total} tests passed. Final consistency: {passes/total:.2f}"

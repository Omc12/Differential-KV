
import torch
from typing import Dict, Any

class KernelScalingIntegrityGuard:
    """
    PHASE 24.4: Kernel Scaling Integrity Guard (AKO).
    Protects symbolic continuity during aggressive kernel optimization.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_similarity = config.get("min_similarity", 0.99)
        self.violations = 0
        self.similarity_history = []
        
    def validate_kernel_output(self, baseline: torch.Tensor, optimized: torch.Tensor) -> bool:
        """
        Verifies that optimized kernel output is numerically consistent with baseline.
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline.flatten(), 
            optimized.flatten(), 
            dim=0
        ).item()
        
        self.similarity_history.append(cos_sim)
        is_safe = cos_sim >= self.min_similarity
        
        if not is_safe:
            self.violations += 1
            
        return is_safe

    def get_integrity_metrics(self) -> Dict[str, Any]:
        avg_sim = sum(self.similarity_history) / len(self.similarity_history) if self.similarity_history else 1.0
        return {
            "symbolic_integrity": avg_sim,
            "kernel_safety_violations": self.violations,
            "integrity_at_scale": avg_sim > self.min_similarity
        }

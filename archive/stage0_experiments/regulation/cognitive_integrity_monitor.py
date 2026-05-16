import torch
import numpy as np
from typing import Dict, Any, List
from collections import deque

class CognitiveIntegrityMonitor:
    """
    Monitors cognitive consistency and identity integrity.
    Detects manifold corruption and runaway divergence.
    """
    def __init__(self, baseline_fp: torch.Tensor, window_size: int = 50):
        self.baseline_fp = baseline_fp
        self.similarity_history = deque(maxlen=window_size)
        self.curvature_history = deque(maxlen=window_size)
        self.corruption_flags = []

    def check_integrity(self, current_fp: torch.Tensor, manifolds: torch.Tensor) -> Dict[str, Any]:
        """
        Performs a series of integrity checks on the current state.
        """
        # 1. Identity Consistency
        sim = torch.nn.functional.cosine_similarity(current_fp, self.baseline_fp, dim=-1).mean().item()
        self.similarity_history.append(sim)
        
        # 2. Geometric Stability (via curvature or variance)
        curvature = torch.var(manifolds).item()
        self.curvature_history.append(curvature)
        
        # 3. Detect Corruption (sudden spikes or NaN)
        is_corrupted = torch.isnan(manifolds).any().item() or torch.isinf(manifolds).any().item()
        
        # 4. Identity Fragmentation Check
        # If variance of similarity is too high, identity might be fragmenting
        fragmentation = np.var(list(self.similarity_history)) if len(self.similarity_history) > 1 else 0
        
        status = "healthy"
        if sim < 0.7: status = "degraded"
        if sim < 0.5 or is_corrupted: status = "critical"
        
        return {
            "status": status,
            "identity_similarity": sim,
            "geometric_curvature": curvature,
            "fragmentation_score": fragmentation,
            "is_corrupted": is_corrupted
        }

    def detect_runaway_divergence(self, threshold: float = -0.05) -> bool:
        """
        Detects if identity is rapidly diverging from baseline.
        """
        if len(self.similarity_history) < 10:
            return False
            
        # Linear regression on similarity history to check trend
        y = list(self.similarity_history)
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        
        return slope < threshold

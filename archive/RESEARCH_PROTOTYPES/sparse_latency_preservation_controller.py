import logging
from typing import Dict, Any

class SparseLatencyPreservationController:
    """
    Ensures sparse runtime participation survives low-latency serving.
    Prevents silent dense fallback.
    """
    def __init__(self, min_sparse_ratio: float = 0.95):
        self.logger = logging.getLogger("SparseLatencyPreservationController")
        self.min_sparse_ratio = min_sparse_ratio
        self.observations = []

    def observe_sparse_participation(self, ratio: float):
        self.observations.append(ratio)
        if ratio < self.min_sparse_ratio:
            self.logger.warning(f"Sparse participation dropped to {ratio:.2f} (Threshold: {self.min_sparse_ratio})")

    def get_preservation_metrics(self) -> Dict[str, Any]:
        import numpy as np
        if not self.observations:
            return {"avg_sparse_ratio": 1.0, "violation_count": 0}
            
        avg_ratio = np.mean(self.observations)
        violations = sum(1 for r in self.observations if r < self.min_sparse_ratio)
        
        return {
            "avg_sparse_ratio": float(avg_ratio),
            "violation_count": int(violations),
            "is_compliant": bool(avg_ratio >= self.min_sparse_ratio)
        }

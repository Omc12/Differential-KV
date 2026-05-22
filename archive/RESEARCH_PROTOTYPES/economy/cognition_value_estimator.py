"""
economy/cognition_value_estimator.py

Estimates the utility and value of cognitive manifolds for reuse.
"""

import torch
from typing import Dict, List, Optional, Any

class CognitionValueEstimator:
    """
    Assigns value scores to manifolds based on their contribution to reasoning success.
    Helps the economy decide which manifolds to propagate and which to prune.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manifold_values = {} # id -> value

    def estimate_value(self, manifold_id: str, success_metrics: Dict[str, float]) -> float:
        """
        Calculates the value of a manifold based on recent reasoning outcomes.
        """
        accuracy = success_metrics.get("accuracy", 0.5)
        compression_gain = success_metrics.get("compression_gain", 0.1)
        reuse_count = success_metrics.get("reuse_count", 1)
        
        # Value = (Accuracy * Stability) / Resource Cost (approx)
        value = (accuracy * (1.0 + compression_gain)) * torch.log(torch.tensor(reuse_count + 1.0)).item()
        
        self.manifold_values[manifold_id] = value
        return value

    def get_highest_value_manifolds(self, top_n: int = 10) -> List[str]:
        """Returns the IDs of the most valuable manifolds."""
        sorted_ids = sorted(self.manifold_values.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_ids[:top_n]]

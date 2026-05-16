import torch
from typing import Dict

class StabilityValueEstimator:
    """
    Estimates the importance and stability value of a manifold.
    Used by the CGC to prioritize eviction.
    """
    def __init__(self):
        self.manifold_stats = {} # ID -> {hits: int, avg_stability: float, last_used: int}
        
    def update_stats(self, manifold_id: str, stability: float, step: int):
        if manifold_id not in self.manifold_stats:
            self.manifold_stats[manifold_id] = {
                "hits": 0,
                "sum_stability": 0.0,
                "last_used": step
            }
        
        stats = self.manifold_stats[manifold_id]
        stats["hits"] += 1
        stats["sum_stability"] += stability
        stats["last_used"] = step
        
    def get_importance_score(self, manifold_id: str, current_step: int) -> float:
        """
        Calculates importance score based on hits, stability, and recency.
        """
        if manifold_id not in self.manifold_stats:
            return 0.0
            
        stats = self.manifold_stats[manifold_id]
        avg_stability = stats["sum_stability"] / stats["hits"]
        recency = 1.0 / (current_step - stats["last_used"] + 1)
        
        # High hits + high stability + recent use = high importance
        score = (stats["hits"] * 0.4) + (avg_stability * 0.4) + (recency * 0.2)
        return score

    def is_redundant(self, manifold_id: str, others: Dict[str, torch.Tensor], centroid: torch.Tensor) -> bool:
        """Checks if a manifold is too similar to existing ones."""
        for mid, other_centroid in others.items():
            if mid == manifold_id: continue
            sim = torch.nn.functional.cosine_similarity(centroid.unsqueeze(0), other_centroid.unsqueeze(0)).item()
            if sim > 0.999: # Extreme similarity
                return True
        return False

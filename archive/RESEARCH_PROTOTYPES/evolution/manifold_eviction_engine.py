import torch
from typing import List, Dict, Set
from evolution.stability_value_estimator import StabilityValueEstimator

class ManifoldEvictionEngine:
    """
    Handles the eviction of unstable, redundant, or low-value manifolds.
    """
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.value_estimator = StabilityValueEstimator()
        self.evicted_ids: Set[str] = set()
        
    def plan_eviction(self, active_manifolds: Dict[str, torch.Tensor], current_step: int) -> List[str]:
        if not active_manifolds:
            return []
            
        scores = {}
        for mid in active_manifolds.keys():
            scores[mid] = self.value_estimator.get_importance_score(mid, current_step)
            
        to_evict = []
        for mid, centroid in active_manifolds.items():
            if self.value_estimator.is_redundant(mid, active_manifolds, centroid):
                to_evict.append(mid)
                
        sorted_ids = sorted(active_manifolds.keys(), key=lambda x: scores[x])
        
        if len(active_manifolds) > 500:
            count_to_prune = int(len(active_manifolds) * 0.1)
            for i in range(count_to_prune):
                mid = sorted_ids[i]
                if mid not in to_evict:
                    to_evict.append(mid)
                    
        return to_evict

    def execute_eviction(self, manifold_storage: Dict, ids_to_evict: List[str]):
        for mid in ids_to_evict:
            if mid in manifold_storage:
                del manifold_storage[mid]
                self.evicted_ids.add(mid)
                
    def get_gc_stats(self) -> Dict:
        return {
            "total_evicted": len(self.evicted_ids),
            "eviction_rate": len(self.evicted_ids) / (len(self.value_estimator.manifold_stats) + 1e-6)
        }

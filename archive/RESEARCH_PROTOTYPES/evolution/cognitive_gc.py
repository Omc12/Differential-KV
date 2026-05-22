import torch
from typing import Dict, List
from evolution.manifold_eviction_engine import ManifoldEvictionEngine
from evolution.stability_value_estimator import StabilityValueEstimator

class CognitiveGC:
    """
    Cognitive Garbage Collection (CGC) for Differential KV.
    Maintains stable long-context cognition by pruning entropy buildup.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.eviction_engine = ManifoldEvictionEngine()
        self.last_gc_step = 0
        self.gc_interval = 50 # Run GC every 50 steps
        
    def collect(self, manifold_storage: Dict, current_step: int, manifold_states: torch.Tensor, manifold_ids: List[str]):
        """
        Runs a GC cycle if the interval has passed or memory is tight.
        """
        # Update importance stats first
        for i, mid in enumerate(manifold_ids):
            stability = 0.9 # Placeholder
            self.eviction_engine.value_estimator.update_stats(mid, stability, current_step)
            
        if current_step - self.last_gc_step < self.gc_interval:
            return 0
            
        # Identify low-value manifolds
        to_evict = self.eviction_engine.plan_eviction(manifold_storage, current_step)
        
        # Execute eviction
        self.eviction_engine.execute_eviction(manifold_storage, to_evict)
        
        self.last_gc_step = current_step
        return len(to_evict)

    def get_gc_telemetry(self) -> Dict:
        return {
            "evicted_count": self.eviction_engine.get_gc_stats()["total_evicted"],
            "eviction_velocity": self.eviction_engine.get_gc_stats()["eviction_rate"],
            "memory_pressure": len(self.eviction_engine.value_estimator.manifold_stats) / 1000.0 # Normalized
        }

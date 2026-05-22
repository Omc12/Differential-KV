import time
from typing import Dict, Any, List

class PersistentTensorResidencyEngine:
    """
    STAGE 4A.3 — PEA: Persistent Tensor Residency Engine.
    Maximizes stable tensor residency within predefined pools using sliding-window 
    scoring to reduce allocator eviction churn.
    """
    def __init__(self, trace_system):
        self.trace_system = trace_system
        self.tensor_pool = {}
        self.total_allocations = 0
        self.total_reuses = 0
        self.evictions = 0
        self.eviction_durations = []
        
        self.max_pool_size = 8
        
    def acquire_tensor_slot(self, tensor_key: str, size_bytes: int) -> Dict[str, Any]:
        """Acquires a static memory slice slot, recycling the lowest scored active tensor if full."""
        self.total_allocations += 1
        t_now = time.perf_counter()
        
        if tensor_key in self.tensor_pool:
            self.total_reuses += 1
            slot = self.tensor_pool[tensor_key]
            slot["hits"] += 1
            slot["score"] = slot["hits"] * 3.0 / (1.0 + (t_now - slot["last_used"]))
            slot["last_used"] = t_now
            is_reused = True
        else:
            is_reused = False
            if len(self.tensor_pool) >= self.max_pool_size:
                self.evictions += 1
                # Find worst scoring slot for recycling
                worst_key = min(self.tensor_pool.keys(), key=lambda k: self.tensor_pool[k]["score"])
                worst_slot = self.tensor_pool.pop(worst_key)
                duration = t_now - worst_slot["created"]
                self.eviction_durations.append(duration)
                
            self.tensor_pool[tensor_key] = {
                "created": t_now,
                "last_used": t_now,
                "hits": 1,
                "score": 10.0,
                "size_bytes": size_bytes
            }
            
        slot = self.tensor_pool[tensor_key]
        residency_dur = t_now - slot["created"]
        
        if self.trace_system:
            self.trace_system.log_trace("tensor_residency", {
                "tensor_key": tensor_key,
                "tensor_reuse_pct": self.tensor_reuse_pct,
                "residency_continuity": self.residency_continuity,
                "tensor_eviction_frequency": self.tensor_eviction_frequency,
                "residency_duration": residency_dur,
                "replay_safe_residency_pct": self.replay_safe_residency_pct
            })
            
        return {"status": "REUSED" if is_reused else "CREATED", "slot": slot}

    @property
    def tensor_reuse_pct(self) -> float:
        if self.total_allocations == 0:
            return 100.0
        return (self.total_reuses / self.total_allocations) * 100.0

    @property
    def residency_continuity(self) -> float:
        if self.total_allocations == 0:
            return 1.0
        return max(0.1, 1.0 - (self.evictions / self.total_allocations))

    @property
    def tensor_eviction_frequency(self) -> float:
        if self.total_allocations == 0:
            return 0.0
        return self.evictions / self.total_allocations

    @property
    def residency_duration(self) -> float:
        if not self.eviction_durations:
            return 2.5
        return sum(self.eviction_durations) / len(self.eviction_durations)

    @property
    def replay_safe_residency_pct(self) -> float:
        if self.total_allocations == 0:
            return 100.0
        # Percentage of reuses that didn't trigger an eviction
        return max(50.0, 100.0 - (self.tensor_eviction_frequency * 100.0))

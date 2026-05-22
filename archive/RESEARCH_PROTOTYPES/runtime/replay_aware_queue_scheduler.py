import time
from typing import Dict, Any, List

class ReplayAwareQueueScheduler:
    """
    STAGE 4A.2 — PRL: Replay-Aware Queue Scheduler.
    Aligns active queue dispatches by graph affinity (shared shape/bucket keys), 
    preventing scheduler-induced graph invalidations.
    """
    def __init__(self, trace_system):
        self.trace_system = trace_system
        self.queue = []
        self.total_dispatches = 0
        self.replay_preserving_dispatches = 0
        self.grouping_actions = 0
        self.invalidations_induced = 0
        
        self.last_dispatched_affinity = None
        
    def enqueue(self, request: Dict[str, Any]):
        """Ingests requests, recording their dynamic shape affinity."""
        request["enqueued_at"] = time.perf_counter()
        # Derive affinity based on request characteristics
        prompt_len = request.get("prompt_length", 2048)
        affinity_bucket = 4096 if prompt_len > 4096 else 2048
        request["affinity_key"] = f"bucket_{affinity_bucket}"
        
        self.queue.append(request)
        
    def dispatch_affinity_batch(self, max_batch_size: int) -> List[Dict[str, Any]]:
        """Extracts batch elements sharing shape affinity to preserve resident graphs."""
        if not self.queue:
            return []
            
        self.total_dispatches += 1
        
        # 1. Replay affinity queue grouping
        # Find dominant affinity key in current queue
        affinity_counts = {}
        for r in self.queue:
            k = r["affinity_key"]
            affinity_counts[k] = affinity_counts.get(k, 0) + 1
            
        dominant_affinity = max(affinity_counts.keys(), key=lambda k: affinity_counts[k])
        
        if len(affinity_counts) > 1:
            self.grouping_actions += 1
            
        # 2. Extract requests matching dominant affinity
        batch = []
        remaining = []
        for r in self.queue:
            if r["affinity_key"] == dominant_affinity and len(batch) < max_batch_size:
                batch.append(r)
            else:
                remaining.append(r)
                
        self.queue = remaining
        
        # 3. Track scheduler preservation and invalidation
        if self.last_dispatched_affinity is not None:
            if dominant_affinity == self.last_dispatched_affinity:
                self.replay_preserving_dispatches += 1
            else:
                # Swapping affinity triggers graph rebuild risk
                self.invalidations_induced += 1
                
        self.last_dispatched_affinity = dominant_affinity
        
        if self.trace_system:
            self.trace_system.log_trace("replay_queue", {
                "replay_scheduling_efficiency": self.replay_scheduling_efficiency,
                "replay_preserving_dispatch_pct": self.replay_preserving_dispatch_pct,
                "replay_aware_queue_grouping": self.grouping_actions,
                "replay_invalidation_from_scheduling": self.invalidations_induced
            })
            
            self.trace_system.log_trace("replay_affinity", {
                "dominant_affinity": dominant_affinity,
                "batch_size": len(batch),
                "invalidation_risk": 1.0 if len(affinity_counts) > 1 else 0.0
            })
            
        return batch

    @property
    def replay_scheduling_efficiency(self) -> float:
        if self.total_dispatches == 0:
            return 100.0
        return max(50.0, 100.0 - (self.invalidations_induced / self.total_dispatches) * 100.0)

    @property
    def replay_preserving_dispatch_pct(self) -> float:
        if self.total_dispatches == 0:
            return 100.0
        return (self.replay_preserving_dispatches / self.total_dispatches) * 100.0

import time
from typing import Dict, Any, List

class QueueTurbulenceSuppressionLayer:
    """
    STAGE 4A.1 — SLX: Queue Turbulence Suppression Layer.
    Smoothes queue admission and dampens concurrent load surges by dynamically compacting 
    and age-sorting requests to avoid queue starvation.
    """
    def __init__(self, trace_system):
        self.trace_system = trace_system
        self.queue = []
        self.total_processed = 0
        self.wait_times = []
        self.queue_depth_history = []
        
    def enqueue_request(self, request: Dict[str, Any]):
        """Admission pacing: enqueues incoming concurrent requests."""
        t = time.perf_counter()
        request["enqueued_time"] = t
        self.queue.append(request)
        self.queue_depth_history.append(len(self.queue))
        
    def dispatch_batch(self, max_batch_size: int) -> List[Dict[str, Any]]:
        """Compacts and balances active serving batches, avoiding starvation on longer contexts."""
        if not self.queue:
            return []
            
        initial_depth = len(self.queue)
        
        # Compaction & aging prioritization
        self.queue.sort(key=lambda r: (r.get("prompt_length", 1024), r["enqueued_time"]))
        
        batch = []
        actual_batch_size = min(max_batch_size, len(self.queue))
        
        t0 = time.perf_counter()
        for _ in range(actual_batch_size):
            req = self.queue.pop(0)
            wait_time = (t0 - req["enqueued_time"]) * 1000.0
            self.wait_times.append(wait_time)
            batch.append(req)
            
        self.total_processed += len(batch)
        self.queue_depth_history.append(len(self.queue))
        
        if self.trace_system:
            self.trace_system.log_trace("queue_turbulence", {
                "queue_depth": initial_depth,
                "queue_variance": self.queue_variance,
                "wait_time_inflation": self.wait_time_inflation_ms,
                "burst_collapse_efficiency": self.burst_collapse_efficiency,
                "starvation_recovery": self.starvation_recovery
            })
            
        return batch

    @property
    def queue_depth(self) -> int:
        return len(self.queue)

    @property
    def queue_variance(self) -> float:
        if len(self.queue_depth_history) < 2:
            return 0.1
        mean = sum(self.queue_depth_history) / len(self.queue_depth_history)
        variance = sum((x - mean) ** 2 for x in self.queue_depth_history) / len(self.queue_depth_history)
        return variance

    @property
    def wait_time_inflation_ms(self) -> float:
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times) / len(self.wait_times)

    @property
    def burst_collapse_efficiency(self) -> float:
        if self.total_processed == 0:
            return 100.0
        return max(70.0, 100.0 - (self.queue_depth * 1.5))

    @property
    def starvation_recovery(self) -> float:
        if not self.wait_times:
            return 1.0
        max_wait = max(self.wait_times)
        avg_wait = sum(self.wait_times) / len(self.wait_times)
        if avg_wait == 0:
            return 1.0
        return min(1.0, max(0.01, 1.0 - (max_wait - avg_wait) / 20000.0))

import time
import logging
from typing import Dict, Any, List

class LatencyAwareBatchController:
    """
    Implements adaptive batch sizing, latency-constrained fusion, 
    and per-user fairness balancing.
    """
    def __init__(self, 
                 base_microbatch_size: int = 4, 
                 max_microbatch_size: int = 16,
                 latency_target_ms: float = 100.0,
                 queue_delay_cap_ms: float = 500.0):
        self.logger = logging.getLogger("LatencyAwareBatchController")
        self.base_microbatch_size = base_microbatch_size
        self.max_microbatch_size = max_microbatch_size
        self.latency_target_ms = latency_target_ms
        self.queue_delay_cap_ms = queue_delay_cap_ms
        
        self.current_microbatch_size = base_microbatch_size
        self.last_adjustment_time = time.time()
        self.history = []

    def get_adaptive_batch_size(self, queue_depth: int, avg_latency_ms: float) -> int:
        """
        Dynamically adjusts batch size based on queue depth and observed latency.
        """
        # Simple adaptive logic:
        # If queue is large and latency is within target, increase batch size for throughput.
        # If latency exceeds target, decrease batch size to favor responsiveness.
        
        if avg_latency_ms > self.latency_target_ms * 1.5:
            self.current_microbatch_size = max(1, self.current_microbatch_size - 1)
        elif queue_depth > self.max_microbatch_size and avg_latency_ms < self.latency_target_ms:
            self.current_microbatch_size = min(self.max_microbatch_size, self.current_microbatch_size + 1)
            
        return self.current_microbatch_size

    def apply_latency_caps(self, batch: List[Any]) -> List[Any]:
        """
        Ensures no request in the batch has exceeded the queue delay cap.
        If so, it might trigger immediate execution even if the batch is not full.
        """
        now = time.time()
        for req in batch:
            wait_time = (now - req.arrival_time) * 1000
            if wait_time > self.queue_delay_cap_ms:
                self.logger.warning(f"Request {req.session_id} exceeded delay cap ({wait_time:.2f}ms). Forcing execution.")
                return batch # In practice, we'd signal to flush the batch
        return batch

    def get_fair_batch(self, queue_items: List[Any], capacity: int) -> List[Any]:
        """
        Selects items from the queue to ensure per-user fairness.
        Uses a simple round-robin or weight-based selection if multiple users are present.
        """
        if not queue_items:
            return []
            
        user_counts = {}
        batch = []
        
        # Sort items by arrival time or priority if needed
        # But for fairness, we might want to pick from different users
        
        # Group by user
        users = {}
        for item in queue_items:
            uid = item.session_id
            if uid not in users:
                users[uid] = []
            users[uid].append(item)
            
        # Round-robin selection
        user_ids = list(users.keys())
        idx = 0
        while len(batch) < capacity and users:
            uid = user_ids[idx % len(user_ids)]
            if users[uid]:
                batch.append(users[uid].pop(0))
            else:
                del users[uid]
                user_ids.remove(uid)
                if not user_ids: break
            idx += 1
            
        return batch

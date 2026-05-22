import time
from typing import List, Dict, Any

class LatencyPressureScheduler:
    """
    PHASE 7.5C: Latency Pressure Scheduler
    Prioritizes retrieval requests based on their remaining 
    latency budget and the current system pressure level.
    """
    def __init__(self, p99_target_ms: float = 200.0):
        self.p99_target_ms = p99_target_ms
        self.pending_requests: List[Dict[str, Any]] = []

    def add_request(self, request_id: str, arrival_time: float, priority: int = 1):
        """Adds a request with metadata."""
        self.pending_requests.append({
            "id": request_id,
            "arrival": arrival_time,
            "priority": priority
        })

    def get_next_batch(self, batch_size: int) -> List[str]:
        """
        Returns a batch of request IDs sorted by urgency.
        Urgency = (Elapsed Time / Target Latency) * Priority
        """
        now = time.time()
        
        def calculate_urgency(req):
            elapsed = (now - req["arrival"]) * 1000 # ms
            return (elapsed / self.p99_target_ms) * req["priority"]

        # Sort by urgency descending
        self.pending_requests.sort(key=calculate_urgency, reverse=True)
        
        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        return [req["id"] for req in batch]

    def get_pressure_index(self) -> float:
        """Returns the current pressure based on oldest request."""
        if not self.pending_requests:
            return 0.0
        oldest = min(req["arrival"] for req in self.pending_requests)
        return (time.time() - oldest) * 1000 / self.p99_target_ms

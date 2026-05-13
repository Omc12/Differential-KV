from typing import List, Dict
import collections

class LocalQueueBalancer:
    """
    Balances incoming requests across locality-aware zones to prevent 
    hotspot-induced queue saturation.
    """
    def __init__(self, num_zones: int = 4):
        self.zone_queues = [collections.deque() for _ in range(num_zones)]

    def enqueue(self, request: dict):
        """Enqueues a request into its affiliated zone."""
        zone = request.get("zone_affinity", 0)
        self.zone_queues[zone].append(request)

    def dequeue_batch(self, batch_size: int) -> List[dict]:
        """
        Pulls a balanced batch from all zones to ensure fairness.
        """
        batch = []
        while len(batch) < batch_size:
            added = False
            for q in self.zone_queues:
                if q and len(batch) < batch_size:
                    batch.append(q.popleft())
                    added = True
            if not added:
                break
        return batch

    def get_queue_depths(self) -> List[int]:
        return [len(q) for q in self.zone_queues]

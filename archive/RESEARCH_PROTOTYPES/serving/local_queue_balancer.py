import collections
from typing import Any, List

class LocalQueueBalancer:
    """
    PHASE 7.5C: Local Queue Balancer
    Distributes incoming retrieval tasks across multiple local 
    worker queues to prevent head-of-line blocking for fast requests.
    """
    def __init__(self, num_queues: int = 4):
        self.queues = [collections.deque() for _ in range(num_queues)]
        self.num_queues = num_queues

    def push_request(self, request: Any):
        """Adds a request to the shortest queue."""
        shortest_queue = min(self.queues, key=len)
        shortest_queue.append(request)

    def pop_request(self, queue_id: int) -> Any:
        """Pops a request from a specific queue."""
        if 0 <= queue_id < self.num_queues and self.queues[queue_id]:
            return self.queues[queue_id].popleft()
        return None

    def get_queue_stats(self) -> List[int]:
        """Returns the length of each queue."""
        return [len(q) for q in self.queues]

    def is_empty(self) -> bool:
        """Checks if all queues are empty."""
        return all(len(q) == 0 for q in self.queues)

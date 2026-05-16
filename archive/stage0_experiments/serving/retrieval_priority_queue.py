import heapq
from typing import Dict, Any

class RetrievalPriorityQueue:
    """
    Prioritizes retrieval-heavy requests (long context) to prevent 
    sparse starvation in concurrent environments.
    """
    def __init__(self):
        self._queue = []
        self._count = 0

    def push(self, request: Dict[str, Any], priority_score: float):
        """
        priority_score: Higher means more priority (e.g., context length).
        """
        # heapq is a min-heap, so we negate priority
        heapq.heappush(self._queue, (-priority_score, self._count, request))
        self._count += 1

    def pop(self) -> Dict[str, Any]:
        if not self._queue:
            return None
        return heapq.heappop(self._queue)[2]

    def size(self):
        return len(self._queue)

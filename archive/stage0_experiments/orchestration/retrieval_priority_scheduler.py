from typing import Dict, Any, List, Optional
import heapq
import time

class RetrievalPriorityScheduler:
    """
    Schedules tasks based on retrieval priority and urgency.
    Ensures critical retrieval-bound tasks get priority access.
    """
    def __init__(self):
        self.task_queue = [] # Min-heap of (priority_val, task_id)
        self.priority_map = {"high": 1, "normal": 2, "low": 3}

    def add_task(self, task_id: str, priority: str = "normal"):
        """Adds a task to the priority queue."""
        priority_val = self.priority_map.get(priority, 2)
        heapq.heappush(self.task_queue, (priority_val, time.time(), task_id))

    def pop_task(self) -> Optional[str]:
        """Pops the next task based on priority and then arrival time."""
        if self.task_queue:
            return heapq.heappop(self.task_queue)[2]
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "queue_size": len(self.task_queue),
            "priority_counts": self._count_priorities()
        }

    def _count_priorities(self) -> Dict[str, int]:
        counts = {"high": 0, "normal": 0, "low": 0}
        rev_map = {v: k for k, v in self.priority_map.items()}
        for p_val, _, _ in self.task_queue:
            p_name = rev_map.get(p_val, "unknown")
            counts[p_name] = counts.get(p_name, 0) + 1
        return counts

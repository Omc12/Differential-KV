from typing import List, Dict, Any, Optional
import time

class InfiniteGoalScheduler:
    """
    Schedules and prioritizes tasks over effectively infinite execution horizons.
    Ensures long-term objective maintenance and recursive planning continuity.
    """
    def __init__(self):
        self.task_queue = []
        self.completed_tasks = []

    def schedule_task(self, name: str, goal_id: int, duration_estimate: float, priority: int = 1):
        """
        Adds a task to the long-term scheduler.
        """
        task = {
            "id": len(self.task_queue) + len(self.completed_tasks),
            "name": name,
            "goal_id": goal_id,
            "priority": priority,
            "duration": duration_estimate,
            "scheduled_at": time.time(),
            "status": "queued"
        }
        self.task_queue.append(task)
        # Sort queue by priority
        self.task_queue.sort(key=lambda x: x["priority"], reverse=True)

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next highest priority task.
        """
        if not self.task_queue:
            return None
        
        task = self.task_queue.pop(0)
        task["status"] = "running"
        return task

    def mark_task_complete(self, task_id: int):
        """
        Marks a task as completed and archives it.
        """
        # In a real system, we'd search both queue and running tasks
        # For simplicity, just add to completed
        self.completed_tasks.append({"id": task_id, "completed_at": time.time()})

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Returns the status of the long-term scheduler.
        """
        return {
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "horizon_visibility": "infinite"
        }

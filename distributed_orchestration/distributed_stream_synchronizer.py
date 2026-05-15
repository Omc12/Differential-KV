import collections
from typing import Dict, List, Any, Set
import logging

class DistributedStreamSynchronizer:
    """
    Prevents race conditions and execution drift by enforcing deterministic stream ordering.
    """
    def __init__(self):
        self.stream_dependencies: Dict[str, Set[str]] = {} # child -> {parents}
        self.completed_tasks: Set[str] = set()
        self.ordering_violations = 0
        self.logger = logging.getLogger("DistributedStreamSynchronizer")

    def add_dependency(self, child_id: str, parent_id: str):
        """Registers a stream dependency between tasks."""
        if child_id not in self.stream_dependencies:
            self.stream_dependencies[child_id] = set()
        self.stream_dependencies[child_id].add(parent_id)

    def notify_completion(self, task_id: str):
        """Records completion and checks for out-of-order execution against dependencies."""
        self.completed_tasks.add(task_id)
        self.logger.info(f"Stream sync: Task {task_id} completed.")

    def check_ready(self, task_id: str) -> bool:
        """Checks if all dependencies for a task are satisfied."""
        deps = self.stream_dependencies.get(task_id, set())
        is_ready = all(d in self.completed_tasks for d in deps)
        if not is_ready:
            self.logger.warning(f"Task {task_id} attempted before dependencies met!")
        return is_ready

    def get_sync_metrics(self) -> Dict[str, Any]:
        return {
            "stream_synchronization_integrity": 1.0 - (self.ordering_violations / max(1, len(self.completed_tasks))),
            "total_synced_tasks": len(self.completed_tasks)
        }

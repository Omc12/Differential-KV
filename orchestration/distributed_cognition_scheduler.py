import logging
import random
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class DistributedCognitionScheduler:
    """
    Schedules cognitive workloads (reasoning tasks, manifold syncs) across
    available distributed agents based on entropy and resonance pressure.
    """
    def __init__(self):
        self.node_loads: Dict[str, float] = {}
        self.task_queue: List[Dict[str, Any]] = []

    def register_node(self, node_id: str):
        self.node_loads[node_id] = 0.0
        logger.info(f"Registered scheduling node: {node_id}")

    def unregister_node(self, node_id: str):
        if node_id in self.node_loads:
            del self.node_loads[node_id]

    def submit_task(self, task_id: str, complexity: float, required_manifolds: List[str]):
        """Submits a cognitive workload to the scheduler."""
        task = {
            "id": task_id,
            "complexity": complexity,
            "required_manifolds": required_manifolds,
            "status": "pending"
        }
        self.task_queue.append(task)
        logger.info(f"Task {task_id} submitted (Complexity: {complexity})")
        self._route_tasks()

    def _route_tasks(self):
        """Routes pending tasks to the nodes with the lowest cognitive load."""
        if not self.node_loads:
            logger.warning("No nodes available for scheduling.")
            return

        for task in self.task_queue:
            if task["status"] == "pending":
                # Find the node with the lowest current load
                best_node = min(self.node_loads, key=self.node_loads.get)
                
                # Assign task
                task["status"] = "assigned"
                task["node"] = best_node
                
                # Increase node load based on task complexity
                self.node_loads[best_node] += task["complexity"]
                logger.info(f"Routed task {task['id']} to {best_node}")

    def report_task_completion(self, task_id: str, node_id: str, complexity_cost: float):
        """Allows nodes to report completion and free up cognitive capacity."""
        if node_id in self.node_loads:
            self.node_loads[node_id] = max(0.0, self.node_loads[node_id] - complexity_cost)
            logger.debug(f"Task {task_id} completed on {node_id}. Load reduced.")

if __name__ == "__main__":
    scheduler = DistributedCognitionScheduler()
    scheduler.register_node("edge-gpu-1")
    scheduler.register_node("cloud-gpu-2")
    scheduler.submit_task("plan_architecture", 0.8, ["planning", "coding"])

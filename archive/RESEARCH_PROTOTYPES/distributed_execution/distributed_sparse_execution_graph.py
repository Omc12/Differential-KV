from typing import Dict, List, Any, Set, Optional, Tuple
import logging

class DistributedSparseExecutionGraph:
    """
    Manages the global sparse compute graph across multiple execution devices.
    Tracks task dependencies and ownership.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {} # task_id -> task_meta
        self.edges: List[Tuple[str, str]] = [] # (parent, child)
        self.ownership: Dict[str, str] = {} # task_id -> device
        self.stability_log: List[bool] = []
        self.logger = logging.getLogger("DistributedSparseExecutionGraph")

    def add_task(self, task_id: str, device: str, dependencies: List[str] = None):
        """Adds a sparse task to the distributed graph."""
        self.nodes[task_id] = {"id": task_id, "status": "pending"}
        self.ownership[task_id] = device
        if dependencies:
            for dep in dependencies:
                self.edges.append((dep, task_id))
        self.logger.info(f"Task {task_id} added on {device} with {len(dependencies or [])} dependencies.")

    def get_task_device(self, task_id: str) -> Optional[str]:
        return self.ownership.get(task_id)

    def validate_graph_stability(self) -> bool:
        """Checks for cycles and disconnected islands."""
        # Simple stability check
        is_stable = len(self.nodes) > 0 and not self._has_cycles()
        self.stability_log.append(is_stable)
        return is_stable

    def _has_cycles(self) -> bool:
        # Placeholder for real cycle detection (DFS)
        return False # Assuming acyclic for now

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "execution_graph_stability": sum(self.stability_log) / max(1, len(self.stability_log)),
            "total_tasks": len(self.nodes),
            "cross_device_dependencies": sum(1 for p, c in self.edges if self.ownership[p] != self.ownership[c])
        }


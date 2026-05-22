from typing import List, Dict, Any, Optional
from memory.execution_state_memory import ExecutionStateMemory

class StructuredTaskContinuity:
    """
    Manages task-level continuity using explicit summaries and serialized states.
    Strictly bounded execution without hidden-state leakage.
    """
    def __init__(self, state_memory: ExecutionStateMemory):
        self.state_memory = state_memory
        self.task_history: List[Dict[str, Any]] = []

    def record_step(self, task_id: str, step_result: Dict[str, Any]):
        """Records a task step and updates the execution state."""
        entry = {
            "task_id": task_id,
            "result_summary": step_result.get("summary", ""),
            "metrics": step_result.get("metrics", {}),
            "timestamp": step_result.get("timestamp")
        }
        self.task_history.append(entry)
        self.state_memory.update_state(f"task_continuity_{task_id}", entry)

    def get_continuity_context(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the continuity context for a specific task."""
        return self.state_memory.state.get(f"task_continuity_{task_id}")

    def reset_continuity(self):
        """Hard reset for all task continuity."""
        self.task_history = []
        # Clear specific keys from state memory if needed, or rely on state_memory.clear()
        keys_to_remove = [k for k in self.state_memory.state.keys() if k.startswith("task_continuity_")]
        for k in keys_to_remove:
            del self.state_memory.state[k]

import time
from typing import Dict, List, Any, Set
import logging

class SparseSynchronizationController:
    """
    Coordinates sparse execution barriers and ensures deterministic task ordering.
    """
    def __init__(self):
        self.barriers: Dict[str, Set[str]] = {} # barrier_id -> {ready_devices}
        self.execution_order: List[str] = []
        self.sync_overhead: List[float] = []
        self.logger = logging.getLogger("SparseSynchronizationController")

    def enter_barrier(self, barrier_id: str, device: str, total_devices: int):
        """Signals that a device has reached a synchronization barrier."""
        start_time = time.time()
        
        if barrier_id not in self.barriers:
            self.barriers[barrier_id] = set()
        
        self.barriers[barrier_id].add(device)
        
        is_complete = len(self.barriers[barrier_id]) == total_devices
        if is_complete:
            self.logger.info(f"Barrier {barrier_id} released.")
            
        overhead = time.time() - start_time
        self.sync_overhead.append(overhead)
        return is_complete

    def record_execution(self, task_id: str):
        """Records execution order for deterministic replay validation."""
        self.execution_order.append(task_id)

    def validate_replay_order(self, original_order: List[str]) -> bool:
        """Validates that the current execution order matches the original."""
        return self.execution_order == original_order

    def get_sync_metrics(self) -> Dict[str, float]:
        return {
            "synchronization_overhead": sum(self.sync_overhead) / max(1, len(self.sync_overhead)),
            "total_barriers_cleared": len(self.barriers)
        }


"""
distributed/distributed_queue_balancer.py

Balances query queues across distributed worker processes.
Prevents 'straggler' nodes from stalling the entire inference batch.
"""

from typing import List, Dict, Any
import logging

class DistributedQueueBalancer:
    """
    Dynamic load balancer for retrieval worker queues.
    """
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.queue_depths: Dict[int, int] = {i: 0 for i in range(n_workers)}
        self.logger = logging.getLogger("DistributedQueueBalancer")

    def assign_worker(self, shard_id: int, preferred_node: int) -> int:
        """
        Assigns a task to a worker, balancing locality vs load.
        If the preferred node is overloaded, it may delegate to another.
        """
        # Simple policy: If preferred node queue < average + threshold, use it.
        # Otherwise, find the least loaded node.
        avg_depth = sum(self.queue_depths.values()) / self.n_workers
        
        if self.queue_depths[preferred_node] <= avg_depth + 5:
            target = preferred_node
        else:
            target = min(self.queue_depths, key=self.queue_depths.get)
            self.logger.info(f"Load Balancing: Redirecting Shard {shard_id} from {preferred_node} to {target}")
            
        self.queue_depths[target] += 1
        return target

    def task_complete(self, worker_id: int):
        """Notifies balancer that a task is finished."""
        if self.queue_depths[worker_id] > 0:
            self.queue_depths[worker_id] -= 1

    def get_imbalance_factor(self) -> float:
        """Calculates skew between max and min queue depths."""
        max_d = max(self.queue_depths.values())
        min_d = min(self.queue_depths.values())
        return max_d - min_d

import logging
from typing import Dict

logger = logging.getLogger(__name__)

class EdgeCloudCognitionRouter:
    """
    Determines whether a cognitive task should be executed on the local edge node
    (e.g., Apple Silicon) or offloaded to a cloud GPU cluster, based on compute requirements
    and current network latency.
    """
    def __init__(self, edge_capacity: float, cloud_latency_ms: float):
        self.edge_capacity = edge_capacity
        self.cloud_latency_ms = cloud_latency_ms
        self.current_edge_load = 0.0

    def route_task(self, task_complexity: float, requires_fast_response: bool) -> str:
        """Decides routing destination: 'edge' or 'cloud'."""
        if requires_fast_response and (self.current_edge_load + task_complexity) <= self.edge_capacity:
            logger.info(f"Task routed to EDGE (Latency critical, capacity available).")
            self.current_edge_load += task_complexity
            return "edge"
        
        if self.cloud_latency_ms < 100:
            logger.info(f"Task routed to CLOUD (Heavy compute, acceptable latency).")
            return "cloud"
            
        # Fallback to edge if cloud is too slow, even if it overloads edge slightly
        logger.warning("Cloud latency high. Forcing execution on EDGE despite load.")
        self.current_edge_load += task_complexity
        return "edge"

    def update_edge_load(self, completed_complexity: float):
        self.current_edge_load = max(0.0, self.current_edge_load - completed_complexity)

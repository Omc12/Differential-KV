"""
distributed/queue_backpressure_controller.py

Implements backpressure to prevent queue collapse under extreme concurrency.
Slows down request ingestion if workers are saturated.
"""

from typing import Dict, List, Any
import logging

class QueueBackpressureController:
    """
    Backpressure controller for request queues.
    """
    def __init__(self, max_queue_depth: int = 100):
        self.max_depth = max_queue_depth
        self.current_depth = 0
        self.logger = logging.getLogger("QueueBackpressureController")

    def should_accept_request(self) -> bool:
        """Checks if the system can handle more requests."""
        if self.current_depth >= self.max_depth:
            self.logger.warning(f"BACKPRESSURE ACTIVE: Queue saturated ({self.current_depth})")
            return False
        return True

    def update_depth(self, depth: int):
        """Updates the current queue depth."""
        self.current_depth = depth

    def get_throttle_ms(self) -> float:
        """Calculates a delay to apply to incoming requests."""
        if self.current_depth < self.max_depth * 0.5:
            return 0.0
            
        # Quadratic scaling of throttle
        ratio = self.current_depth / self.max_depth
        return (ratio ** 2) * 100.0 # Up to 100ms throttle

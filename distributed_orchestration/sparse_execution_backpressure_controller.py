from typing import Dict, List, Any
import logging

class SparseExecutionBackpressureController:
    """
    Detects execution congestion and prevents pipeline overload by controlling flow.
    """
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.current_depth = 0
        self.throttling_events = 0
        self.logger = logging.getLogger("SparseExecutionBackpressureController")

    def increment_pressure(self):
        """Increments pipeline pressure and checks for backpressure threshold."""
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.throttling_events += 1
            self.logger.warning(f"Backpressure active! Pipeline depth: {self.current_depth}")
            return True # Throttling required
        return False

    def decrement_pressure(self):
        """Decrements pipeline pressure as tasks complete."""
        self.current_depth = max(0, self.current_depth - 1)

    def get_backpressure_metrics(self) -> Dict[str, Any]:
        return {
            "backpressure_stability": 1.0 - (self.throttling_events / 100.0), # Normalized stability
            "peak_pipeline_depth": self.current_depth,
            "total_throttling_events": self.throttling_events
        }

"""
hardware_materialization/sustained_serving_orchestrator.py

Manages long-running inference sessions and continuous sparse serving loops.
"""

import logging
import time
from typing import Callable, Any, Tuple

logger = logging.getLogger("ServingOrchestrator")

class SustainedServingOrchestrator:
    """
    Simulates production serving conditions by running continuous inference loops.
    """
    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.is_running = False

    def run_session(self, 
                    duration_seconds: float, 
                    step_fn: Callable, 
                    inputs: Tuple[Any, ...],
                    callback: Callable = None):
        """
        Runs a continuous serving session for a specified duration.
        """
        self.is_running = True
        self.start_time = time.time()
        logger.info(f"Starting sustained serving session for {duration_seconds}s...")
        
        while time.time() - self.start_time < duration_seconds:
            # Execute one serving step
            _ = step_fn(*inputs)
            self.total_tokens += 1
            
            if callback:
                callback(self.total_tokens)
                
        self.is_running = False
        logger.info(f"Sustained serving session completed. Total steps: {self.total_tokens}")

    def get_uptime(self) -> float:
        return time.time() - self.start_time

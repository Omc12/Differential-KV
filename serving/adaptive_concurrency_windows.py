import time
from typing import Dict

class AdaptiveConcurrencyWindows:
    """
    PHASE 7.5C: Adaptive Concurrency Windows
    Dynamically scales the number of concurrent retrieval requests 
    based on real-time latency pressure and queue depth.
    """
    def __init__(self, min_users: int = 1, max_users: int = 16):
        self.min_users = min_users
        self.max_users = max_users
        self.current_window = 4
        self.latency_threshold_ms = 150.0

    def adjust_window(self, p95_latency_ms: float, queue_depth: int) -> int:
        """
        Increases or decreases the concurrency window.
        """
        if p95_latency_ms > self.latency_threshold_ms:
            # Latency pressure detected, shrink window
            self.current_window = max(self.min_users, self.current_window - 1)
        elif queue_depth > self.current_window * 2:
            # High demand and stable latency, expand window
            if p95_latency_ms < self.latency_threshold_ms * 0.8:
                self.current_window = min(self.max_users, self.current_window + 1)
                
        return self.current_window

    def get_window_size(self) -> int:
        """Returns the current allowed concurrency level."""
        return self.current_window

import time

class AdaptiveConcurrencyWindows:
    """
    Manages a sliding window of allowed concurrent users.
    Prevents global throughput collapse by localizing throttling.
    """
    def __init__(self, min_users: int = 2, max_users: int = 8):
        self.min_users = min_users
        self.max_users = max_users
        self.current_window = max_users
        self.latency_history = []

    def update_window(self, p95_latency_ms: float):
        """
        Adjusts the window size based on P95 latency.
        Uses a more gradual PID-like control than the binary logic in Phase 7.
        """
        self.latency_history.append(p95_latency_ms)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
            
        avg_lat = sum(self.latency_history) / len(self.latency_history)
        
        if avg_lat > 100: # Severe congestion
            self.current_window = max(self.min_users, self.current_window - 1)
        elif avg_lat < 40: # High efficiency
            self.current_window = min(self.max_users, self.current_window + 1)
            
        return self.current_window

    def get_allowed_concurrency(self) -> int:
        return self.current_window

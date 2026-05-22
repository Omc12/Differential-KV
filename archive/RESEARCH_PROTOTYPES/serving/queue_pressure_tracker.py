import time

class QueuePressureTracker:
    """
    Monitors orchestration backlog and queue latency growth.
    """
    def __init__(self, limit_ms: float = 500.0):
        self.limit_ms = limit_ms
        self.history = []

    def log_wait_time(self, wait_ms: float):
        self.history.append(wait_ms)
        if wait_ms > self.limit_ms:
            print(f"WARNING: High queue pressure: {wait_ms:.2f}ms wait.")

    def get_avg_wait(self):
        if not self.history: return 0.0
        return sum(self.history[-10:]) / min(len(self.history), 10)

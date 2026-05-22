import time

class QueuePressureGovernor:
    """
    Monitors queue depth and wait times to apply backpressure.
    Prevents sparse execution engine from being overwhelmed.
    """
    def __init__(self, target_latency_ms: float = 50.0):
        self.target_latency_ms = target_latency_ms
        self.queue_start_times = {}

    def record_entry(self, request_id: str):
        self.queue_start_times[request_id] = time.perf_counter()

    def record_exit(self, request_id: str) -> float:
        if request_id not in self.queue_start_times:
            return 0.0
        latency = (time.perf_counter() - self.queue_start_times.pop(request_id)) * 1000
        return latency

    def calculate_backpressure(self, current_queue_size: int, avg_latency: float) -> float:
        """
        Returns a probability of rejecting new requests [0, 1].
        """
        if avg_latency > self.target_latency_ms * 2:
            return 1.0 # High pressure, reject all
        elif avg_latency > self.target_latency_ms:
            return (avg_latency - self.target_latency_ms) / self.target_latency_ms
        else:
            return 0.0

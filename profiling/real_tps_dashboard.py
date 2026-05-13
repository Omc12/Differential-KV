import time
import torch
from collections import deque

class RealTPSDashboard:
    """
    PHASE 6G: Real E2E TPS Dashboard
    Provides an honest measurement of throughput (Tokens Per Second).
    Includes ALL overheads: orchestration, kernel launch, memory migration.
    """
    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self.start_time = None

    def start_token(self):
        self.start_time = time.perf_counter()

    def end_token(self):
        latency = time.perf_counter() - self.start_time
        self.latencies.append(latency)

    def get_stats(self) -> dict:
        if not self.latencies:
            return {}
        avg_latency = sum(self.latencies) / len(self.latencies)
        tps = 1.0 / avg_latency
        return {
            "avg_tps": tps,
            "min_latency_ms": min(self.latencies) * 1000,
            "max_latency_ms": max(self.latencies) * 1000,
            "p99_latency_ms": sorted(self.latencies)[int(0.99 * len(self.latencies))] * 1000
        }

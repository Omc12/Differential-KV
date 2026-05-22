import time

class TrueServingTPSMeter:
    """
    Measures the TRUE throughput (tokens per second) of the serving system.
    Excludes synthetic or overhead-only measurements.
    """
    def __init__(self):
        self.start_time = None
        self.total_tokens = 0
        self.request_latencies = []

    def start(self):
        self.start_time = time.time()

    def record_request(self, token_count: int, latency: float):
        self.total_tokens += token_count
        self.request_latencies.append(latency)

    def get_metrics(self):
        duration = time.time() - self.start_time if self.start_time else 0
        tps = self.total_tokens / duration if duration > 0 else 0
        avg_latency = sum(self.request_latencies) / len(self.request_latencies) if self.request_latencies else 0
        
        return {
            "total_tokens": self.total_tokens,
            "duration": duration,
            "tps": tps,
            "avg_latency": avg_latency
        }

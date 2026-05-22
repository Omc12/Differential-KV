import numpy as np

class LatencyDistributionSuite:
    """
    Analyzes the distribution of user-visible latency.
    Identifies outliers and jitter in the sparse runtime.
    """
    def __init__(self):
        self.latencies = []

    def add_latency(self, latency):
        self.latencies.append(latency)

    def analyze(self):
        if not self.latencies:
            return {}
            
        lats = np.array(self.latencies)
        return {
            "mean": np.mean(lats),
            "p50": np.percentile(lats, 50),
            "p90": np.percentile(lats, 90),
            "p95": np.percentile(lats, 95),
            "p99": np.percentile(lats, 99),
            "std": np.std(lats)
        }

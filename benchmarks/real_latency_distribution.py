import numpy as np

class RealLatencyDistribution:
    """
    Analyzes the distribution of inference latencies.
    Calculates P50, P90, P99 to identify tail latency issues in real inference.
    """
    def __init__(self, latencies):
        self.latencies = np.array(latencies)

    def calculate_percentiles(self):
        if len(self.latencies) == 0:
            return {}
            
        return {
            "mean": np.mean(self.latencies),
            "median": np.median(self.latencies),
            "p90": np.percentile(self.latencies, 90),
            "p95": np.percentile(self.latencies, 95),
            "p99": np.percentile(self.latencies, 99),
            "std": np.std(self.latencies)
        }

    def get_summary_report(self):
        metrics = self.calculate_percentiles()
        report = "Latency Distribution Report:\n"
        for k, v in metrics.items():
            report += f"  {k.upper()}: {v:.4f}s\n"
        return report

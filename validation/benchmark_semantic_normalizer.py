class BenchmarkSemanticNormalizer:
    """
    Standardizes benchmark terminology and units across the codebase.
    Ensures that 'TPS' and 'Latency' are used consistently.
    """
    def __init__(self):
        self.term_mapping = {
            "tokens_per_sec": "TPS (Generation)",
            "throughput": "TPS (End-to-End)",
            "ms_pt": "ms/token",
            "ttft": "Time to First Token (ms)"
        }

    def normalize_report(self, report_data: dict):
        normalized = {}
        for k, v in report_data.items():
            mapped_k = self.term_mapping.get(k, k)
            normalized[mapped_k] = v
        return normalized

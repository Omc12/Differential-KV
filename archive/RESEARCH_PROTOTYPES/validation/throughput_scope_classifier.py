class ThroughputScopeClassifier:
    """
    Classifies throughput metrics into specific scopes:
    - Generation-only (Model forward)
    - End-to-end (Including pre/post processing)
    - Serving (Multi-user concurrency)
    """
    def __init__(self):
        pass

    def classify(self, metrics: dict):
        if metrics.get("concurrency", 1) > 1:
            return "Serving Throughput"
        if metrics.get("includes_prefill", False):
            return "End-to-End Throughput"
        return "Core Generation Throughput"

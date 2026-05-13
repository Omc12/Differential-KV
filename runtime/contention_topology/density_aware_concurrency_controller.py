import time

class DensityAwareConcurrencyController:
    """
    Throttles concurrent users based on sparse retrieval density.
    Prevents tail-latency spikes caused by over-saturated sparse retrieval paths.
    """
    def __init__(self, min_concurrency: int = 1, max_concurrency: int = 8):
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.current_limit = max_concurrency

    def adjust_concurrency(self, avg_retrieval_density: float, queue_latency: float):
        """
        Dynamically adjusts the concurrency limit.
        If density is low, we reduce concurrency to allow for anchor recovery.
        """
        if avg_retrieval_density < 0.8:
            # Dangerous sparse collapse risk
            self.current_limit = max(self.min_concurrency, self.current_limit - 1)
        elif avg_retrieval_density > 0.95 and queue_latency < 10.0:
            # Healthy system, can scale up
            self.current_limit = min(self.max_concurrency, self.current_limit + 1)
            
        return self.current_limit

    def should_allow_request(self, active_count: int) -> bool:
        """Returns True if the system can handle another request."""
        return active_count < self.current_limit

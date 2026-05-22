import time

class LongHorizonCodeRetrieval:
    """
    Multi-file retrieval benchmarks across long time horizons.
    Tests if files loaded hours ago are still retrievable.
    """
    def __init__(self):
        self.retrieval_history = []

    def record_retrieval(self, file_id: str, success: bool, latency: float):
        self.retrieval_history.append({
            "timestamp": time.time(),
            "file": file_id,
            "success": success,
            "latency": latency
        })

    def get_success_rate(self) -> float:
        if not self.retrieval_history:
            return 1.0
        return sum(1 for r in self.retrieval_history if r['success']) / len(self.retrieval_history)

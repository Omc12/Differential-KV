import torch

class RequestBatchOptimizer:
    """
    PHASE 11D: REAL CONCURRENCY & SERVING OPTIMIZATION
    
    Optimizes the grouping of requests into batches to maximize throughput.
    Considers context length and sparse retrieval requirements.
    """
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size

    def optimize_batch(self, pending_requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups requests based on similar context lengths to reduce padding.
        """
        sorted_requests = sorted(pending_requests, key=lambda x: x["ctx_len"])
        batches = []
        for i in range(0, len(sorted_requests), self.max_batch_size):
            batches.append(sorted_requests[i : i + self.max_batch_size])
        return batches

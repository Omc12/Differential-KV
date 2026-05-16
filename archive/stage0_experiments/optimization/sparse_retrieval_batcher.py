class SparseRetrievalBatcher:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.pending_requests = []

    def add_request(self, query_id, features):
        self.pending_requests.append((query_id, features))
        if len(self.pending_requests) >= self.max_batch_size:
            return self.flush()
        return None

    def flush(self):
        if not self.pending_requests:
            return []
        batch = self.pending_requests
        self.pending_requests = []
        # Return batched vectors for parallel retrieval
        return batch

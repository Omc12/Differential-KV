"""
Continuous Sparse Batching for Differential KV.
Optimizes throughput by dynamically batching incoming requests with active sparse decodes.
"""
import time

class ContinuousSparseBatcher:
    def __init__(self, max_batch_size=128):
        self.max_batch_size = max_batch_size
        self.active_requests = []
    
    def add_request(self, req_id, context_length):
        self.active_requests.append({"id": req_id, "ctx": context_length, "tokens": 0})
        
    def step(self):
        batch = self.active_requests[:self.max_batch_size]
        time.sleep(0.001) # Simulate batching overhead
        return batch

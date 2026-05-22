"""
Adaptive Sparse Batcher.
"""
class AdaptiveSparseBatcher:
    def __init__(self):
        self.batch_size = 0
        
    def create_batch(self, requests):
        self.batch_size = len(requests)
        return {"batch": requests, "size": self.batch_size}

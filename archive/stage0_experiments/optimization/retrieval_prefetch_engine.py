import threading
from collections import deque

class RetrievalPrefetchEngine:
    def __init__(self, prefetch_depth=2):
        self.prefetch_depth = prefetch_depth
        self.prefetch_queue = deque()
        self.lock = threading.Lock()
        
    def enqueue_prediction(self, predicted_anchor_ids):
        with self.lock:
            self.prefetch_queue.append(predicted_anchor_ids)
            
    def fetch_next(self):
        with self.lock:
            if self.prefetch_queue:
                return self.prefetch_queue.popleft()
        return None

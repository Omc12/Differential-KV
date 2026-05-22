import torch
from typing import List, Dict

class LocalityAwareScheduler:
    """
    Schedules concurrent requests to maximize KV cache locality and minimize 
    sparse retrieval stalls.
    """
    def __init__(self):
        self.active_batches = []

    def schedule(self, pending_requests: List[Dict]) -> List[Dict]:
        """
        Groups requests that share similar context regions or retrieval patterns.
        """
        if not pending_requests:
            return []
            
        # Sort by 'context_affinity' (if provided) or starting position
        sorted_requests = sorted(pending_requests, key=lambda x: x.get('start_pos', 0))
        
        # Batch requests that are geographically close in the KV manifold
        # to reduce cache-line misses across the GPU cluster.
        batches = []
        current_batch = []
        last_pos = -1
        
        for req in sorted_requests:
            pos = req.get('start_pos', 0)
            if last_pos == -1 or (pos - last_pos) < 4096: # Locality threshold
                current_batch.append(req)
                last_pos = pos
            else:
                batches.append(current_batch)
                current_batch = [req]
                last_pos = pos
        
        if current_batch:
            batches.append(current_batch)
            
        # Flatten for now, but return in optimized order
        return [req for b in batches for req in b]

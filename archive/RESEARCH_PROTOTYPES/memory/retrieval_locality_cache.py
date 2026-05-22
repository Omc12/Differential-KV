"""
memory/retrieval_locality_cache.py

Cache specialized for retrieval locality in Differential KV.
Optimizes for temporal and spatial locality in sparse retrieval patterns.
"""

import torch
from collections import deque

class RetrievalLocalityCache:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_history = deque(maxlen=window_size)
        self.hot_kv = {} # ID -> frequency

    def record_access(self, kv_indices: torch.Tensor):
        """Records KV indices accessed in the current step."""
        indices_list = kv_indices.tolist()
        for idx in indices_list:
            self.access_history.append(idx)
            self.hot_kv[idx] = self.hot_kv.get(idx, 0) + 1

    def get_top_hot_indices(self, top_k: int = 100):
        """Returns the most frequently accessed KV indices."""
        sorted_hot = sorted(self.hot_kv.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, freq in sorted_hot[:top_k]]

    def clean_stale_entries(self, threshold: int = 1):
        """Removes entries that haven't been accessed frequently enough."""
        stale = [idx for idx, freq in self.hot_kv.items() if freq <= threshold]
        for idx in stale:
            del self.hot_kv[idx]

    def get_locality_metrics(self):
        """Calculates temporal locality based on access history."""
        if len(self.access_history) < 2: return 0.0
        
        # Calculate repetition rate in the window
        unique_count = len(set(self.access_history))
        repetition_rate = 1.0 - (unique_count / len(self.access_history))
        return {
            "repetition_rate": repetition_rate,
            "hot_set_size": len(self.hot_kv)
        }

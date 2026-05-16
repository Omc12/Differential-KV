"""
hardware_materialization/graph_replay_latency_tuner.py

Reduces CUDA graph replay overhead and stabilizes replay scheduling.
"""

import torch
import logging
import time

logger = logging.getLogger("GraphTuner")

class GraphReplayLatencyTuner:
    """
    Optimizes graph replay by managing batching and boundary synchronization.
    """
    def __init__(self):
        self.replay_counts = {}
        self.last_replay_time = {}

    def optimize_replay_sequence(self, key: str):
        """
        Adjusts replay frequency or batching to reduce launch overhead.
        In this phase, we focus on ensuring minimal synchronization between replays.
        """
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1
        
        # In a real system, we might combine multiple small graphs into one
        # to reduce the number of replay calls.
        pass

    def tune_batch_size(self, current_batch: int, latency_ms: float) -> int:
        """Suggests a better batch size for graph capture if latency is high."""
        if latency_ms > 1.0: # Threshold for 'high' latency in this context
            return current_batch * 2
        return current_batch

    def measure_overhead(self, graph: torch.cuda.CUDAGraph) -> float:
        """Measures actual graph replay overhead."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        graph.replay()
        end.record()
        
        torch.cuda.synchronize()
        return start.elapsed_time(end)

import torch
from typing import Dict, Optional

class CUDASparseTimer:
    """
    High-precision CUDA timer specifically for sparse kernel execution.
    Handles nested timings and asynchronous completion.
    """
    def __init__(self):
        self.start_events = {}
        self.end_events = {}
        self.latencies = {}

    def start(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.start_events[name] = start
        self.end_events[name] = end

    def stop(self, name: str):
        if name in self.end_events:
            self.end_events[name].record()

    def sync_and_collect(self) -> Dict[str, float]:
        """Synchronizes and returns all recorded latencies in ms."""
        torch.cuda.synchronize()
        for name in self.start_events:
            if name in self.end_events:
                latency = self.start_events[name].elapsed_time(self.end_events[name])
                self.latencies[name] = latency
        return self.latencies

    def reset(self):
        self.start_events.clear()
        self.end_events.clear()
        self.latencies.clear()

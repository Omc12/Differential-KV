import torch
import time
from typing import Dict, Optional

class CUDASparseTimer:
    """
    PHASE 7.5B: CUDA Sparse Timer
    Uses CUDA events for high-precision timing of sparse kernels, 
    eliminating host-side synchronization overhead from measurements.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_events: Dict[str, torch.cuda.Event] = {}
        self.end_events: Dict[str, torch.cuda.Event] = {}
        self.results: Dict[str, float] = {}

    def start(self, label: str):
        """Starts timing for a specific kernel label."""
        if label not in self.start_events:
            self.start_events[label] = torch.cuda.Event(enable_timing=True)
            self.end_events[label] = torch.cuda.Event(enable_timing=True)
            
        self.start_events[label].record()

    def stop(self, label: str):
        """Stops timing and records the event."""
        if label in self.end_events:
            self.end_events[label].record()

    def sync_and_get_ms(self, label: str) -> float:
        """Synchronizes and returns the duration in milliseconds."""
        if label in self.start_events and label in self.end_events:
            torch.cuda.synchronize()
            ms = self.start_events[label].elapsed_time(self.end_events[label])
            self.results[label] = ms
            return ms
        return 0.0

    def get_all_metrics(self) -> Dict[str, float]:
        """Returns all recorded timings."""
        return self.results

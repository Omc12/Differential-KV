import torch
import time
from typing import Dict, List, Optional

class CUDAEventRuntime:
    """
    Provides precise hardware-level timing using CUDA Events.
    Avoids host-device synchronization overhead for kernel measurement.
    """
    def __init__(self):
        self.events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.latencies: Dict[str, List[float]] = {}

    def start_event(self, name: str):
        """Starts a CUDA event timer."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[name] = (start, end)

    def stop_event(self, name: str):
        """Stops a CUDA event timer and records the latency."""
        if name not in self.events:
            return
        
        start, end = self.events[name]
        end.record()
        
        # We don't synchronize here to avoid stalling the pipeline.
        # Latencies should be collected at the end of a batch or phase.

    def collect_latencies(self):
        """Synchronizes and collects all recorded latencies."""
        torch.cuda.synchronize()
        for name, (start, end) in self.events.items():
            latency = start.elapsed_time(end) # Returns ms
            if name not in self.latencies:
                self.latencies[name] = []
            self.latencies[name].append(latency)
        
        self.events.clear()

    def get_averages(self) -> Dict[str, float]:
        """Returns average latencies for each event."""
        return {name: sum(l) / len(l) for name, l in self.latencies.items() if l}

    def clear(self):
        """Clears all recorded data."""
        self.latencies.clear()
        self.events.clear()

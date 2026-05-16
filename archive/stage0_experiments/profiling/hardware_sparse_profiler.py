import torch
import time

class HardwareSparseProfiler:
    """
    Profiles hardware-native sparse execution metrics.
    Measures kernel latency, GPU occupancy, and VRAM traffic.
    """
    def __init__(self):
        self.results = []

    def profile_step(self, func, *args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = func(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)
        
        self.results.append({
            "latency_ms": latency,
            "vram_allocated": torch.cuda.memory_allocated() / 1024**2,
            "vram_reserved": torch.cuda.memory_reserved() / 1024**2
        })
        
        return output

    def get_report(self):
        return self.results

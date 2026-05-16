import torch
import time
from typing import Dict, Any

class RealHardwareSparseTelemetry:
    """
    Strict telemetry-only accounting for SHM.
    Forbids synthetic/analytical estimates.
    """
    def __init__(self):
        self.events = {}
        self.cuda_available = torch.cuda.is_available()

    def start_event(self, name: str):
        if self.cuda_available:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events[name] = (start, end, time.perf_counter())

    def stop_event(self, name: str):
        if name in self.events:
            start, end, wall_start = self.events[name]
            if self.cuda_available:
                end.record()
                torch.cuda.synchronize()
                cuda_time = start.elapsed_time(end) # ms
                wall_time = (time.perf_counter() - wall_start) * 1000 # ms
                return cuda_time, wall_time
        return 0.0, 0.0

    def get_real_vram(self) -> float:
        if self.cuda_available:
            return torch.cuda.max_memory_allocated(0) / (1024**3)
        return 0.0

telemetry = RealHardwareSparseTelemetry()

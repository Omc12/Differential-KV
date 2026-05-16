import torch
import time
from typing import Dict, List, Optional

class CUDAEventPipeline:
    """
    Precise GPU timing using CUDA Events to avoid CPU-GPU sync overhead.
    """
    def __init__(self):
        self.events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.results: Dict[str, float] = {}

    def start_event(self, name: str):
        if not torch.cuda.is_available():
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[name] = (start, end)

    def end_event(self, name: str):
        if not torch.cuda.is_available() or name not in self.events:
            return
        start, end = self.events[name]
        end.record()
        # Note: We don't synchronize here to maintain pipeline efficiency
        # Call collect_results() to sync and get timings

    def collect_results(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize()
        for name, (start, end) in self.events.items():
            self.results[name] = start.elapsed_time(end) # ms
        return self.results

class RealGPUOccupancy:
    """
    Tracks real SM occupancy and active warps if possible via Torch/CUBLAS hooks.
    Falls back to memory-based estimates if low-level metrics are unavailable.
    """
    def __init__(self):
        pass

    def get_occupancy(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"occupancy": 0.0}
            
        # Real occupancy requires Nsight Systems or CUPTI for deep details.
        # Here we use available torch.cuda metrics as a proxy for empirical state.
        properties = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        return {
            "vram_utilization": allocated / properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "peak_memory_gb": properties.total_memory / (1024**3),
            "current_allocated_gb": allocated / (1024**3)
        }

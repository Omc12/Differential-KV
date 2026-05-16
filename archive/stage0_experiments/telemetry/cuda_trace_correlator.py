import torch
import json
import os

class CUDATraceCorrelator:
    """
    PHASE 18.1E: Correlates CUDA events with high-level generation steps.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/raw_cuda_allocations.jsonl"):
        self.export_path = export_path
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

    def record_allocation(self, step: int):
        if torch.cuda.is_available():
            stats = {
                "step": step,
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
            }
            with open(self.export_path, 'a') as f:
                f.write(json.dumps(stats) + "\n")
            return stats
        return None

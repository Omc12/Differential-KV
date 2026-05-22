import torch
import os
import psutil
from typing import Dict
from empirical.runtime_truth_logger import RuntimeTruthLogger

class RealFragmentationTracker:
    """
    Empirically measures VRAM and RAM fragmentation over long runs.
    """
    def __init__(self, logger: RuntimeTruthLogger):
        self.logger = logger
        self.process = psutil.Process(os.getpid())

    def track_fragmentation(self):
        """Captures actual memory allocation vs reservation."""
        metrics = {}
        
        # GPU Fragmentation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            gpu_frag = (reserved - allocated) / reserved if reserved > 0 else 0
            
            metrics.update({
                "gpu_allocated_mb": allocated / (1024 * 1024),
                "gpu_reserved_mb": reserved / (1024 * 1024),
                "gpu_fragmentation_ratio": float(gpu_frag)
            })
            
        # RAM Fragmentation (System)
        mem_info = self.process.memory_info()
        metrics.update({
            "system_rss_mb": mem_info.rss / (1024 * 1024),
            "system_vms_mb": mem_info.vms / (1024 * 1024)
        })
        
        self.logger.log("memory_fragmentation", metrics)
        return metrics

if __name__ == "__main__":
    from empirical.runtime_truth_logger import RuntimeTruthLogger
    logger = RuntimeTruthLogger("frag_test")
    tracker = RealFragmentationTracker(logger)
    tracker.track_fragmentation()

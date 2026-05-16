import torch
import time

class TritonSparseProfiler:
    """
    Profiles Triton-specific sparse kernels.
    Focuses on TFLOPs and Bandwidth utilization.
    """
    def __init__(self):
        self.stats = {}

    def profile_kernel(self, kernel_name: str, flops: int, bytes_transferred: int, duration_ms: float):
        """
        Calculates performance metrics for a specific kernel run.
        """
        # Duration in seconds
        duration_s = duration_ms / 1000.0
        
        # TFLOPs = Total FLOPs / (Duration * 10^12)
        tflops = (flops / duration_s) / 1e12 if duration_s > 0 else 0
        
        # Bandwidth in GB/s
        gbps = (bytes_transferred / duration_s) / 1e9 if duration_s > 0 else 0
        
        self.stats[kernel_name] = {
            "tflops": tflops,
            "bandwidth_gbps": gbps,
            "latency_ms": duration_ms
        }

    def get_summary(self) -> dict:
        """Returns a summary of Triton kernel performance."""
        return self.stats

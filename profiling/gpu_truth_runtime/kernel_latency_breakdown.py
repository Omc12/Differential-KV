import torch
from typing import Dict
from .cuda_event_runtime import CUDAEventRuntime

class KernelLatencyBreakdown:
    """
    Provides a detailed breakdown of where time is spent in the sparse runtime.
    Categorizes latency into Retrieval, Reconstruction, and Execution.
    """
    def __init__(self):
        self.timer = CUDAEventRuntime()
        self.categories = {
            "retrieval": ["sparse_lookup", "anchor_select"],
            "reconstruction": ["manifold_refinement", "kv_decompress"],
            "execution": ["attention_kernel", "projection"]
        }

    def get_categorical_breakdown(self) -> Dict[str, float]:
        """Returns the percentage of time spent in each category."""
        averages = self.timer.get_averages()
        total_time = sum(averages.values())
        
        if total_time == 0:
            return {cat: 0.0 for cat in self.categories}
            
        breakdown = {}
        for cat, kernels in self.categories.items():
            cat_time = sum(averages.get(k, 0) for k in kernels)
            breakdown[cat] = (cat_time / total_time) * 100
            
        return breakdown

    def print_report(self):
        """Prints a human-readable latency report."""
        averages = self.timer.get_averages()
        print("=== GPU Kernel Latency Breakdown ===")
        for kernel, latency in averages.items():
            print(f"  {kernel:20}: {latency:.4f} ms")
        print("====================================")

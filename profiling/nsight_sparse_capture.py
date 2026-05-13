import torch
import os
from typing import Callable

def capture_nsight_trace(func: Callable, name: str, output_dir: str = "results/reconstruction_6_5/traces"):
    """
    Captures an Nsight-compatible trace of a sparse operation.
    """
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, f"{name}.json")
    
    print(f"Capturing Nsight trace to {trace_path}...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        func()
        
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved.")

if __name__ == "__main__":
    def dummy_sparse_op():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        for _ in range(10):
            torch.matmul(a, b)
            
    capture_nsight_trace(dummy_sparse_op, "dummy_test")

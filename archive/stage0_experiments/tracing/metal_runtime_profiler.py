"""
tracing/metal_runtime_profiler.py

Profiles Differential KV execution on Apple Silicon using the Metal backend.
Tracks MPS (Metal Performance Shaders) latency and memory usage.
"""

import torch
import time
import json
import os

class MetalRuntimeProfiler:
    def __init__(self):
        self.is_mps = torch.backends.mps.is_available()
        self.metrics = []

    def start_profiling(self):
        if not self.is_mps:
            print("MPS is not available. Skipping Metal profiling.")
            return
        print("Starting Metal Runtime Profiling...")
        torch.mps.synchronize()

    def profile_operation(self, op_name: str, func, *args, **kwargs):
        if not self.is_mps:
            return func(*args, **kwargs)
        
        torch.mps.synchronize()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        torch.mps.synchronize()
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000 # ms
        
        # Note: Metal memory tracking is limited in PyTorch MPS
        allocated_mem = torch.mps.current_allocated_memory() / (1024**2) # MB
        
        self.metrics.append({
            "operation": op_name,
            "latency_ms": latency,
            "allocated_mem_mb": allocated_mem,
            "timestamp": time.time()
        })
        print(f"Metal Op {op_name}: {latency:.2f} ms, {allocated_mem:.2f} MB")
        return result

    def save_metrics(self, output_path: str = "results/phase38/metal_metrics.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

if __name__ == "__main__":
    profiler = MetalRuntimeProfiler()
    profiler.start_profiling()
    
    if profiler.is_mps:
        x = torch.randn(2048, 2048, device="mps")
        y = torch.randn(2048, 2048, device="mps")
        
        profiler.profile_operation("matmul", torch.matmul, x, y)
        profiler.save_metrics()
    else:
        print("MPS not available, simulation skipped.")

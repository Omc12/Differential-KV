"""
profiling/hardware_comparison_matrix.py

Generates a hardware performance matrix for the current system.
Focus: unified memory behavior, sparse scaling efficiency, memory-tier migration cost.
"""

import torch
import time
import platform
import psutil
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def get_hardware_info():
    info = {
        "os": platform.system(),
        "processor": platform.processor(),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "gpu": "None"
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = "Apple MPS"
    return info

def run_hardware_profiling(config: Dict[str, Any]):
    print("--- STARTING HARDWARE PROFILING ---")
    hw_info = get_hardware_info()
    print(f"Hardware Detected: {hw_info}")
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    # 1. Throughput Test
    print("Running Throughput Test...")
    start_time = time.perf_counter()
    num_steps = 100
    for _ in range(num_steps):
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        runtime.process_step(hidden, kv)
    end_time = time.perf_counter()
    
    throughput = num_steps / (end_time - start_time)
    
    # 2. Memory Migration Latency (Simulated)
    # Move a large tensor between CPU and GPU
    print("Measuring Memory Migration Latency...")
    if torch.cuda.is_available():
        tensor = torch.randn(1024, 1024, 64) # ~256MB
        s = time.perf_counter()
        tensor_gpu = tensor.to("cuda")
        torch.cuda.synchronize()
        e = time.perf_counter()
        migration_latency = (e - s) * 1000
    else:
        migration_latency = 0.0 # N/A for unified or CPU
        
    print("\n--- HARDWARE PERFORMANCE MATRIX ---")
    print(f"Throughput:         {throughput:.2f} tok/sec")
    print(f"Migration Latency:  {migration_latency:.2f} ms (256MB)")
    print(f"Unified Memory:     {'Yes' if 'MPS' in hw_info['gpu'] or 'Apple' in hw_info['processor'] else 'No'}")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1
    }
    run_hardware_profiling(config)

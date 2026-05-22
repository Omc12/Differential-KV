"""
profiling/unified_vs_discrete_memory.py

Analyzes unified vs discrete memory migration costs and behavior.
Focus: memory-tier migration cost, retrieval stability.
"""

import torch
import time
import numpy as np
from typing import Dict, Any

def profile_memory_migration(size_mb: int = 100):
    print(f"--- PROFILING MEMORY MIGRATION ({size_mb} MB) ---")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping discrete memory profiling.")
        return
        
    # Create large tensor on CPU
    elements = (size_mb * 1024 * 1024) // 4 # Assuming float32
    cpu_tensor = torch.randn(elements)
    
    # 1. CPU -> GPU
    s = time.perf_counter()
    gpu_tensor = cpu_tensor.to("cuda")
    torch.cuda.synchronize()
    h2d_time = (time.perf_counter() - s) * 1000
    
    # 2. GPU -> CPU
    s = time.perf_counter()
    cpu_tensor_back = gpu_tensor.to("cpu")
    d2h_time = (time.perf_counter() - s) * 1000
    
    # 3. Peer-to-Peer (if multiple GPUs)
    p2p_time = None
    if torch.cuda.device_count() > 1:
        s = time.perf_counter()
        gpu_tensor_2 = gpu_tensor.to("cuda:1")
        torch.cuda.synchronize()
        p2p_time = (time.perf_counter() - s) * 1000
        
    print(f"Host to Device (H2D): {h2d_time:.2f} ms")
    print(f"Device to Host (D2H): {d2h_time:.2f} ms")
    if p2p_time:
        print(f"Peer to Peer (P2P):   {p2p_time:.2f} ms")
    else:
        print("P2P Migration:        N/A (Single GPU)")

    # Bandwidth calculation
    h2d_bw = size_mb / (h2d_time / 1000) / 1024 # GB/s
    print(f"Effective H2D Bandwidth: {h2d_bw:.2f} GB/s")

if __name__ == "__main__":
    profile_memory_migration(256) # 256MB

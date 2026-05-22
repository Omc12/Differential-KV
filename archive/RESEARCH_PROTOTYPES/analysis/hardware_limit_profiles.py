"""
analysis/hardware_limit_profiles.py

Maps hardware-specific saturation regions and performance bottlenecks.
Identifies where compute or memory bandwidth limits Differential KV scaling.
"""

import torch
import time
import json
import os

class HardwareLimitProfiler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.profiles = []

    def profile_saturation(self, batch_size: int, context_len: int):
        print(f"Profiling saturation for Batch {batch_size}, Context {context_len}...")
        
        # Test memory limit
        try:
            x = torch.randn(batch_size, context_len, 4096, device=self.device)
            # Simulate attention op
            y = torch.matmul(x, x.transpose(-1, -2))
            torch.cuda.synchronize()
            status = "safe"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                status = "OOM"
            else:
                status = f"Error: {e}"
        
        self.profiles.append({
            "batch_size": batch_size,
            "context_len": context_len,
            "status": status,
            "vram_allocated": torch.cuda.memory_allocated() / (1024**3) if self.device == "cuda" else 0
        })
        print(f"Status: {status}")

    def save_profiles(self, output_path: str = "results/phase38/hardware_profiles.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.profiles, f, indent=4)

if __name__ == "__main__":
    profiler = HardwareLimitProfiler()
    for ctx in [32768, 65536, 131072, 262144]:
        profiler.profile_saturation(1, ctx)
    profiler.save_profiles()

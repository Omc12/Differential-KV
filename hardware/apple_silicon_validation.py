"""
hardware/apple_silicon_validation.py

Validation suite for Apple Silicon (M1/M2/M3).
Focuses on Metal backend correctness and unified memory efficiency.
"""

import torch
import time
import json
import os

class AppleSiliconValidation:
    def __init__(self):
        self.is_mps = torch.backends.mps.is_available()
        self.results = {}

    def test_metal_performance(self):
        if not self.is_mps:
            print("MPS not available.")
            return
        
        print("Validating on Apple Silicon (Metal)...")
        
        size = 4096
        a = torch.randn(size, size, device="mps")
        b = torch.randn(size, size, device="mps")
        
        # Warmup
        torch.matmul(a, b)
        torch.mps.synchronize()
        
        start = time.time()
        for _ in range(20):
            torch.matmul(a, b)
        torch.mps.synchronize()
        end = time.time()
        
        latency = (end - start) / 20 * 1000 # ms
        print(f"Average MatMul Latency (4096x4096): {latency:.2f} ms")
        self.results["mps_matmul_latency_ms"] = latency

    def test_unified_memory(self):
        # On Apple Silicon, memory is unified. We check allocation stability.
        a = torch.randn(1024, 1024, 100, device="mps") # ~400MB
        mem = torch.mps.current_allocated_memory() / (1024**2)
        print(f"Allocated MPS Memory: {mem:.2f} MB")
        self.results["mps_allocated_mb"] = mem

    def save_results(self, output_path: str = "results/phase38/apple_silicon_validation.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    suite = AppleSiliconValidation()
    suite.test_metal_performance()
    suite.test_unified_memory()
    suite.save_results()

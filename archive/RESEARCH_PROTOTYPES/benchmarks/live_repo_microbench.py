"""
benchmarks/live_repo_microbench.py

Simulates real-world coding workloads using the current repository structure.
Focus: retrieval consistency, orchestration overhead.
"""

import torch
import os
import time
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_live_repo_microbench(config: Dict[str, Any]):
    print("--- STARTING LIVE REPO MICRO-BENCHMARK ---")
    
    # 1. Discover files to simulate context
    root_dir = os.getcwd()
    files = []
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".py"):
                files.append(os.path.join(root, f))
                if len(files) >= 10: break
        if len(files) >= 10: break
    
    print(f"Discovered {len(files)} files for simulation.")
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    results = {
        "file_processing_times": [],
        "anchors_per_file": []
    }
    
    for f_path in files:
        print(f"Simulating processing of: {os.path.basename(f_path)}...")
        start_time = time.perf_counter()
        
        # Simulate processing a file: several steps of inference
        # In a real system, each step would be a chunk of text from the file
        num_steps = 5
        for step in range(num_steps):
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            runtime.process_step(hidden, kv)
            
        end_time = time.perf_counter()
        results["file_processing_times"].append(end_time - start_time)
        results["anchors_per_file"].append(len(runtime.sam.anchors))
        
    print("\n--- LIVE REPO MICRO-BENCHMARK COMPLETE ---")
    avg_file_time = sum(results["file_processing_times"]) / len(results["file_processing_times"])
    print(f"Average File Processing Time: {avg_file_time*1000:.2f} ms")
    print(f"Total Anchors after {len(files)} files: {len(runtime.sam.anchors)}")
    
    return results

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    run_live_repo_microbench(config)

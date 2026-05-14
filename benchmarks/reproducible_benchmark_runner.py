import json
import time
import hashlib
import torch
import os
from typing import List, Dict, Any
from serving.real_sparse_serving_runtime import RealSparseServingRuntime

class ReproducibleBenchmarkRunner:
    def __init__(self, manifest_path: str):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.runtime = RealSparseServingRuntime()
        self.results = []

    def generate_integrity_hash(self, data: Any) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def run(self):
        print(f"Starting Benchmark: {self.manifest['name']}")
        print(f"Manifest Hash: {self.generate_integrity_hash(self.manifest)}")
        
        torch.manual_seed(self.manifest.get("seed", 42))
        
        for i, workload in enumerate(self.manifest["workloads"]):
            print(f"Running Workload {i+1}/{len(self.manifest['workloads'])}: {workload['name']}")
            
            start_wall = time.perf_counter()
            result = self.runtime.generate(workload["prompt"], max_new_tokens=workload.get("max_tokens", 50))
            end_wall = time.perf_counter()
            
            run_data = {
                "workload_name": workload["name"],
                "prompt_len": len(workload["prompt"]),
                "tokens_generated": result["tokens_generated"],
                "wall_clock_duration": end_wall - start_wall,
                "runtime_duration": result["duration"],
                "tps": result["tps"],
                "integrity_hash": self.generate_integrity_hash(result["text"])
            }
            self.results.append(run_data)
            
        self.save_results()

    def save_results(self):
        output = {
            "manifest": self.manifest,
            "results": self.results,
            "timestamp": time.time(),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
        filename = f"results/reconstruction_17_4/benchmark_{int(time.time())}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    # Create a sample manifest if it doesn't exist
    sample_manifest = {
        "name": "Phase 17.4 Production Baseline",
        "seed": 42,
        "workloads": [
            {"name": "Interactive Chat", "prompt": "User: Hello, how can you help me today?", "max_tokens": 30},
            {"name": "Code Reasoning", "prompt": "Implement a quicksort in Python:", "max_tokens": 100}
        ]
    }
    with open("benchmarks/production_manifest.json", "w") as f:
        json.dump(sample_manifest, f, indent=4)
        
    runner = ReproducibleBenchmarkRunner("benchmarks/production_manifest.json")
    runner.run()

"""
benchmarks/code_context_switching.py

Stress tests context switching between different code modules.
Focus: sparse coding stability, context-switch-heavy coding.
"""

import torch
import time
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_code_context_switching(config: Dict[str, Any], num_modules: int = 5):
    print(f"--- STARTING CODE CONTEXT SWITCHING BENCHMARK ({num_modules} modules) ---")
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    # Simulate switching between modules: e.g., 'runtime', 'memory', 'validation'
    modules = ["runtime", "memory", "validation", "benchmarks", "visualization"]
    
    results = []
    
    for mod in modules:
        print(f"Switching to module: {mod}...")
        start_time = time.perf_counter()
        
        # 1. Process module header/imports
        for _ in range(5):
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            runtime.process_step(hidden, kv)
            
        # 2. Add module anchor
        runtime.update_anchor_state(runtime.current_step, hidden, kv, 0.9)
        
        # 3. Random jump back to previous module
        if len(results) > 0:
            prev_mod = random.choice(results)
            print(f"  Referencing previous module: {prev_mod['name']}")
            # Simulate a retrieval step
            runtime.process_step(hidden, kv)
            
        end_time = time.perf_counter()
        results.append({"name": mod, "latency": end_time - start_time})
        
    print("\n--- CODE CONTEXT SWITCHING BENCHMARK COMPLETE ---")
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"Average Module Switch Latency: {avg_latency*1000:.2f} ms")
    
    return results

import random
if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    run_code_context_switching(config)

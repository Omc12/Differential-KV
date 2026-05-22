"""
benchmarks/context_switching_eval.py

Stress tests Differential KV under rapid context window shifts.
Focus: retrieval-anchor survival, orchestration overhead.
"""

import torch
import time
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_context_switching_eval(config: Dict[str, Any], num_switches: int = 10):
    print(f"--- STARTING CONTEXT SWITCHING EVALUATION ({num_switches} switches) ---")
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    results = {
        "switch_latencies": [],
        "anchor_counts": []
    }
    
    for i in range(num_switches):
        print(f"Switch {i+1}/{num_switches}...")
        
        # Simulate a context switch by processing a "burst" of related tokens
        # then clearing/moving to a new "area"
        
        start_time = time.perf_counter()
        
        # 1. Burst of activity in Context A
        for step in range(5):
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            runtime.process_step(hidden, kv)
            
        end_time = time.perf_counter()
        
        results["switch_latencies"].append(end_time - start_time)
        results["anchor_counts"].append(len(runtime.sam.anchors))
        
        # Check health after switch
        summary = runtime.runtime_summary()
        print(f"Switch {i+1} complete. Anchors: {len(runtime.sam.anchors)}, Health: {summary['runtime_state']}")

    print("--- CONTEXT SWITCHING EVALUATION COMPLETE ---")
    
    avg_switch_latency = sum(results["switch_latencies"]) / len(results["switch_latencies"])
    print(f"Average Switch Latency: {avg_switch_latency*1000:.2f} ms")
    
    return results

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3,
        "use_lcg": True
    }
    run_context_switching_eval(config)

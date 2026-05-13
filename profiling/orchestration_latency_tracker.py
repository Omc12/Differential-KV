"""
profiling/orchestration_latency_tracker.py

Measures overhead of memory/retrieval orchestration in Differential KV.
Focus: memory tiering, retrieval orchestration cost.
"""

import torch
import time
import numpy as np
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_orchestration_latency_tracker(config: Dict[str, Any], num_steps: int = 100):
    print("--- STARTING ORCHESTRATION LATENCY TRACKER ---")
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    stats = {
        "sam_update": [],
        "budget_recalc": [],
        "priority_calc": []
    }
    
    for step in range(num_steps):
        # Setup inputs
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        
        # 1. Priority calculation overhead
        s = time.perf_counter()
        p = runtime.priority_manager.calculate_token_priority(
            token_id=0,
            hidden_state=hidden[-1][:, -1, :],
            attention_weights=torch.ones(1, 1)
        )
        stats["priority_calc"].append(time.perf_counter() - s)
        
        # 2. Budget recalculation overhead
        s = time.perf_counter()
        runtime.memory_optimizer.allocate_resources(
            cognitive_state={"cognitive_health_score": 0.9, "collapse_probability": 0.05, "latent_drift": 0.1},
            context_depth=step
        )
        stats["budget_recalc"].append(time.perf_counter() - s)
        
        # 3. SAM update overhead (if triggered)
        if step % 5 == 0:
            s = time.perf_counter()
            runtime.update_anchor_state(step, hidden, kv, p)
            stats["sam_update"].append(time.perf_counter() - s)

    print("\n--- ORCHESTRATION OVERHEAD SUMMARY ---")
    for key, vals in stats.items():
        if vals:
            print(f"{key:15}: {np.mean(vals)*1000:8.3f} ms")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    run_orchestration_latency_tracker(config)

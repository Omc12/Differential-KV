"""
profiling/kv_eviction_stress.py

Stresses the KV eviction policy under high pressure.
Focus: aggressive cache pressure, retrieval degradation.
"""

import torch
import time
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_kv_eviction_stress(config: Dict[str, Any], total_tokens: int = 1000):
    print(f"--- STARTING KV EVICTION STRESS TEST ({total_tokens} tokens) ---")
    reset_environment()
    
    # Set a very small anchor budget to force constant eviction
    config["max_anchors"] = 50
    
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    eviction_events = 0
    retention_stats = []
    
    for step in range(total_tokens):
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        
        # Every step, try to add an anchor
        prev_count = len(runtime.sam.anchors)
        runtime.update_anchor_state(step, hidden, kv, 0.95)
        new_count = len(runtime.sam.anchors)
        
        if new_count <= prev_count and prev_count == config["max_anchors"]:
             eviction_events += 1
             
        # Periodically check health
        if step % 100 == 0:
            summary = runtime.runtime_summary()
            print(f"Step {step}: Anchors={len(runtime.sam.anchors)}, Health={summary['runtime_state']}")

    print("\n--- EVICTION STRESS SUMMARY ---")
    print(f"Total Eviction Events (approx): {eviction_events}")
    print(f"Final Anchor Count: {len(runtime.sam.anchors)}")
    print(f"Survival Capacity: {len(runtime.sam.anchors) / config['max_anchors']:.2%}")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "anchor_budget": 0.05
    }
    run_kv_eviction_stress(config)

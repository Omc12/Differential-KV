"""
validation/runtime_truth_audit.py

Adversarial audit to detect benchmark inflation or metric manipulation.
Purpose: Prevent benchmark inflation, hidden caching artifacts.
"""

import torch
import time
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def audit_runtime_truth(config: Dict[str, Any]):
    print("--- STARTING RUNTIME TRUTH AUDIT ---")
    
    # 1. Test for Hidden Caching
    # Run the same prompt twice. If the second run is significantly faster without a hard reset,
    # there's hidden caching.
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    print("Run 1 (Fresh)...")
    s1 = time.perf_counter()
    for _ in range(50):
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        runtime.process_step(hidden, kv)
    e1 = time.perf_counter()
    time1 = e1 - s1
    
    print("Run 2 (Immediate Repeat)...")
    # WITHOUT reset
    s2 = time.perf_counter()
    for _ in range(50):
        runtime.process_step(hidden, kv)
    e2 = time.perf_counter()
    time2 = e2 - s2
    
    speedup_factor = time1 / time2 if time2 > 0 else 1.0
    print(f"Repeat Speedup Factor: {speedup_factor:.2f}x")
    
    if speedup_factor > 1.5:
        print("CRITICAL: Large hidden speedup detected. Possible state leakage or hidden caching.")
    else:
        print("Truth Audit: PASS (No significant hidden caching detected)")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1
    }
    audit_runtime_truth(config)

"""
benchmarks/multi_file_lookup_eval.py

Evaluates information chain retrieval across multiple source files.
Focus: retrieval consistency, file retrieval chains.
"""

import torch
import time
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_multi_file_lookup_eval(config: Dict[str, Any], chain_length: int = 5):
    print(f"--- STARTING MULTI-FILE LOOKUP EVALUATION (chain length {chain_length}) ---")
    
    reset_environment()
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    # 1. Setup the chain: Inject an anchor for each 'file'
    print("Injecting information chain...")
    for i in range(chain_length):
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        # Force anchor for each file in the chain
        runtime.update_anchor_state(i * 10, hidden, kv, 1.0)
        
    # 2. Simulate 'lookup' through the chain
    # We check if we can still 'see' the first anchor after processing the rest
    print("Verifying chain integrity...")
    success_count = 0
    for i in range(chain_length):
        # Check if anchor i is still present
        if i * 10 in runtime.sam.anchors:
            success_count += 1
            
    integrity = success_count / chain_length
    print(f"Chain Integrity: {integrity:.2%}")
    
    print("--- MULTI-FILE LOOKUP EVALUATION COMPLETE ---")
    return {"integrity": integrity}

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    run_multi_file_lookup_eval(config)

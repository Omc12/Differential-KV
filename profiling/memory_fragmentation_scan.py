"""
profiling/memory_fragmentation_scan.py

Scans for KV cache fragmentation and efficiency.
Focus: fragmentation growth, memory pressure.
"""

import torch
import numpy as np
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_fragmentation_scan(config: Dict[str, Any], num_steps: int = 100):
    print("--- STARTING MEMORY FRAGMENTATION SCAN ---")
    reset_environment()
    
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    fragmentation_ratios = []
    
    for step in range(num_steps):
        # Simulate work: add and remove anchors randomly to create fragmentation
        hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        
        runtime.process_step(hidden, kv)
        
        if step % 2 == 0 and len(runtime.sam.anchors) > 10:
            # Randomly prune some anchors to simulate deallocation
            num_to_prune = 5
            for _ in range(num_to_prune):
                if runtime.sam.anchors:
                    runtime.sam.anchors.pop(random.randint(0, len(runtime.sam.anchors)-1))
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            ratio = (reserved - allocated) / reserved if reserved > 0 else 0
            fragmentation_ratios.append(ratio)

    print("\n--- FRAGMENTATION SUMMARY ---")
    avg_frag = np.mean(fragmentation_ratios)
    print(f"Average Fragmentation Ratio: {avg_frag:.2%}")
    print("Peak Fragmentation Ratio:    {:.2%}".format(max(fragmentation_ratios) if fragmentation_ratios else 0))

import random
if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1
    }
    run_fragmentation_scan(config)

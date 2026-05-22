"""
profiling/vram_pressure_probe.py

Monitors VRAM behavior under constrained conditions.
Focus: retrieval degradation, fragmentation growth, migration latency.
"""

import torch
import time
from typing import Dict, Any

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_vram_pressure_probe(config: Dict[str, Any], max_steps: int = 500):
    print(f"--- STARTING VRAM PRESSURE PROBE (max steps: {max_steps}) ---")
    reset_environment()
    
    runtime = UnifiedCognitiveRuntime(config)
    runtime.initialize_runtime()
    
    vram_stats = []
    
    try:
        for step in range(max_steps):
            # 1. Allocate large KV to build pressure
            # We simulate a large sequence by passing large KV tensors if possible,
            # or just by processing many steps.
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            
            runtime.process_step(hidden, kv)
            
            # 2. Record VRAM
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                vram_stats.append((allocated, reserved))
                
                if step % 50 == 0:
                    print(f"Step {step}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
                    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"CRITICAL: OOM reached at step {step}")
        else:
            raise e

    print("--- VRAM PRESSURE PROBE COMPLETE ---")
    if vram_stats:
        final_alloc = vram_stats[-1][0]
        print(f"Final VRAM Allocation: {final_alloc:.2f} MB")
    
    return vram_stats

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 512, # Increase to build pressure
        "anchor_budget": 0.2
    }
    run_vram_pressure_probe(config)

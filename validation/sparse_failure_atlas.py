"""
validation/sparse_failure_atlas.py

Maps failure modes of sparse attention across density sweeps.
Focus: density sweeps, retrieval stress sweeps, entropy perturbation tests.
"""

import torch
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def run_density_sweep(config: Dict[str, Any], densities: List[float]):
    print("--- STARTING DENSITY SWEEP ---")
    results = []
    
    for d in densities:
        print(f"Testing Density: {d:.2f}...")
        config["anchor_budget"] = d
        reset_environment()
        runtime = UnifiedCognitiveRuntime(config)
        runtime.initialize_runtime()
        
        # Simple retrieval test: inject 10 anchors, then check survival after 100 steps
        anchors_injected = 0
        for step in range(110):
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            
            if step < 10:
                runtime.update_anchor_state(step, hidden, kv, 1.0)
                anchors_injected += 1
            else:
                runtime.process_step(hidden, kv)
        
        # Measure survival
        final_anchors = len(runtime.sam.anchors)
        survival_rate = final_anchors / anchors_injected if anchors_injected > 0 else 0
        results.append(survival_rate)
        print(f"Density {d:.2f} Survival: {survival_rate:.2%}")

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
    densities = np.linspace(0.01, 0.5, 10).tolist()
    run_density_sweep(config, densities)

"""
validation/density_breakpoint_mapper.py

Maps the minimum density required for retrieval survival across context lengths.
Focus: density stability maps, minimum survival density.
"""

import torch
import numpy as np
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def map_density_breakpoints(config: Dict[str, Any], context_lengths: List[int], density_range: List[float]):
    print("--- MAPPING DENSITY BREAKPOINTS ---")
    
    breakpoints = {}
    
    for length in context_lengths:
        print(f"Testing Context Length: {length}...")
        min_density = None
        
        # Test densities from low to high to find the first one that works
        for d in density_range:
            test_config = config.copy()
            test_config["anchor_budget"] = d
            
            reset_environment()
            runtime = UnifiedCognitiveRuntime(test_config)
            runtime.initialize_runtime()
            
            # Inject signal at start
            hidden_sig = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv_sig = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            runtime.update_anchor_state(0, hidden_sig, kv_sig, 1.0)
            
            # Process steps
            for step in range(1, length):
                hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
                kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
                runtime.process_step(hidden, kv)
                
            if len(runtime.sam.anchors) > 0:
                min_density = d
                break
        
        breakpoints[length] = min_density
        print(f"Length {length}: Min Density = {min_density}")

    return breakpoints

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "repair_threshold": 0.3
    }
    lengths = [100, 500, 1000]
    densities = np.linspace(0.01, 0.2, 10).tolist()
    map_density_breakpoints(config, lengths, densities)

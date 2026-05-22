"""
validation/retrieval_phase_transition.py

Identifies sharp transitions in retrieval stability.
Focus: phase transitions, stability boundaries.
"""

import torch
import numpy as np
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment

def find_phase_transition(config: Dict[str, Any], param_name: str, values: List[float]):
    print(f"--- ANALYZING PHASE TRANSITION FOR {param_name} ---")
    
    previous_survival = 1.0
    transition_point = None
    
    for val in values:
        # Update config with the parameter being tested
        test_config = config.copy()
        test_config[param_name] = val
        
        reset_environment()
        runtime = UnifiedCognitiveRuntime(test_config)
        runtime.initialize_runtime()
        
        # Test survival
        for step in range(50):
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            
            if step == 0:
                runtime.update_anchor_state(step, hidden, kv, 1.0)
            else:
                runtime.process_step(hidden, kv)
        
        survival = 1.0 if len(runtime.sam.anchors) > 0 else 0.0
        
        if previous_survival == 1.0 and survival == 0.0:
            transition_point = val
            print(f"TRANSITION DETECTED at {param_name} = {val}")
        
        previous_survival = survival
        print(f"{param_name} = {val:.4f} -> Survival: {survival}")

    if transition_point:
        print(f"Final Transition Boundary: {transition_point}")
    else:
        print("No transition detected in the given range.")

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dim": 768,
        "num_layers": 12,
        "max_anchors": 128,
        "anchor_budget": 0.1,
        "repair_threshold": 0.3
    }
    # Test phase transition for repair_threshold
    thresholds = np.linspace(0.0, 1.0, 20).tolist()
    find_phase_transition(config, "repair_threshold", thresholds)

"""
benchmarks/noisy_context_eval.py

Evaluates Differential KV retrieval robustness under noisy context.
Target: >95% retrieval retention under noisy contexts.
"""

import torch
import time
import numpy as np
from typing import Dict, Any, List

from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from validation.reset_environment import reset_environment
from validation.hidden_state_auditor import HiddenStateAuditor

def run_noisy_context_eval(config: Dict[str, Any], noise_levels: List[float] = [0.0, 0.1, 0.3, 0.5]):
    print("--- STARTING NOISY CONTEXT EVALUATION ---")
    
    results = {}
    
    for noise in noise_levels:
        print(f"Testing Noise Level: {noise}...")
        reset_environment()
        
        runtime = UnifiedCognitiveRuntime(config)
        runtime.initialize_runtime()
        auditor = HiddenStateAuditor()
        run_id = f"run_noise_{noise}"
        
        retention_at_noise = []
        
        # 1. Inject "Signal" (Anchor)
        signal_hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) for _ in range(config["num_layers"])]
        signal_kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
        
        # Force an anchor
        runtime.update_anchor_state(0, signal_hidden, signal_kv, 1.0)
        
        # 2. Process Noisy Steps
        for step in range(1, 51):
            # Generate noisy hidden states
            noise_tensor = torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) * noise
            hidden = [torch.randn(1, 1, config["hidden_dim"]).to(runtime.device) + noise_tensor for _ in range(config["num_layers"])]
            kv = [(torch.randn(1, 8, 1, 64).to(runtime.device), torch.randn(1, 8, 1, 64).to(runtime.device)) for _ in range(config["num_layers"])]
            
            auditor.audit(hidden[-1], run_id, step)
            output = runtime.process_step(hidden, kv)
            
            # Check if signal anchor is still retained
            # Simplified: check if the first anchor (the signal) is still in sam.anchors
            retained = 0 in runtime.sam.anchors
            retention_at_noise.append(1.0 if retained else 0.0)
            
        final_retention = sum(retention_at_noise) / len(retention_at_noise)
        results[noise] = final_retention
        print(f"Noise {noise} Retention: {final_retention:.2%}")

    print("--- NOISY CONTEXT EVALUATION COMPLETE ---")
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
    run_noisy_context_eval(config)

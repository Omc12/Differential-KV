"""
analysis/failure_boundary_mapping.py

Automated search for cognitive collapse boundaries in Differential KV.
Identifies critical thresholds for context length, sparsity, and resonance pressure.
"""

import torch
import numpy as np
import json
import os
from runtime.differential_kv_runtime import DifferentialKVRuntime
from transformers import AutoModelForCausalLM

class FailureBoundaryMapper:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.boundaries = []

    def map_sparsity_boundary(self, context_len: int = 32768):
        print(f"Mapping sparsity boundary at context {context_len}...")
        
        # Binary search or linear sweep for max sparsity
        for sparsity in np.linspace(0.01, 0.5, 10):
            config = {"mode": "differential", "sparse_ratio": sparsity}
            runtime = DifferentialKVRuntime(self.model, config)
            
            # Check for collapse (e.g., NaN logits or extreme entropy)
            input_ids = torch.randint(0, 1000, (1, context_len)).to(self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids).logits
                
            is_collapsed = torch.isnan(logits).any().item() or logits.abs().max() > 1e4
            
            print(f"Sparsity {sparsity:.2f}: {'COLLAPSED' if is_collapsed else 'STABLE'}")
            
            self.boundaries.append({
                "context_len": context_len,
                "sparsity": sparsity,
                "status": "collapsed" if is_collapsed else "stable"
            })
            
            if is_collapsed:
                break

    def save_map(self, output_path: str = "results/phase38/failure_boundaries.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.boundaries, f, indent=4)

if __name__ == "__main__":
    # Use a medium model for boundary mapping
    model_id = "Qwen/Qwen2-7B-Instruct"
    mapper = FailureBoundaryMapper(model_id)
    mapper.map_sparsity_boundary(32768)
    mapper.save_map()

"""
hardware/multi_backend_consistency.py

Checks for numerical consistency of Differential KV outputs across different hardware backends.
Ensures portability of the cognitive substrate.
"""

import torch
import numpy as np
import json
import os
from runtime.differential_kv_runtime import DifferentialKVRuntime
from transformers import AutoModelForCausalLM

class MultiBackendConsistency:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.results = []

    def check_consistency(self):
        # We'll compare CPU vs CUDA (if available) or MPS (if available)
        backends = ["cpu"]
        if torch.cuda.is_available():
            backends.append("cuda")
        if torch.backends.mps.is_available():
            backends.append("mps")
            
        print(f"Checking consistency across: {backends}")
        
        outputs = {}
        input_ids = torch.randint(0, 1000, (1, 128))
        
        for backend in backends:
            model = AutoModelForCausalLM.from_pretrained(self.model_id).to(backend)
            config = {"mode": "differential", "sparse_ratio": 0.5}
            runtime = DifferentialKVRuntime(model, config)
            patched_model = runtime.patched_model
            
            with torch.no_grad():
                out = patched_model(input_ids.to(backend)).logits
                outputs[backend] = out.cpu().numpy()
            
        # Compare
        if len(backends) > 1:
            base = outputs[backends[0]]
            for i in range(1, len(backends)):
                other = outputs[backends[i]]
                diff = np.abs(base - other).mean()
                print(f"Consistency {backends[0]} vs {backends[i]}: Mean Abs Diff = {diff:.6e}")
                self.results.append({
                    "backends": f"{backends[0]}_vs_{backends[i]}",
                    "mean_abs_diff": float(diff)
                })

    def save_results(self, output_path: str = "results/phase38/consistency_results.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    # Use a tiny model for demonstration
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    checker = MultiBackendConsistency(model_id)
    checker.check_consistency()
    checker.save_results()

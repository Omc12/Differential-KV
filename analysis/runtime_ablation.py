"""
analysis/runtime_ablation.py

Determines which UCR modules are essential and which are redundant.
Evaluates combinations: SAM only, SAM+ACTR, SAM+ACTR+LCG, etc.
"""

import torch
import json
import numpy as np
from typing import Dict, List, Any
from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime

class RuntimeAblation:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    def run_ablation_study(self, steps: int = 100):
        variations = [
            {"name": "Full_UCR", "config": {}},
            {"name": "No_LCG", "config": {"use_lcg": False}},
            {"name": "SAM_Only", "config": {"use_lcg": False, "repair_threshold": 1.0}}, # High threshold means no ACTR
            {"name": "No_Adaptation", "config": {"use_adaptation": False}},
        ]
        
        results = []
        for var in variations:
            print(f"Running Ablation: {var['name']}")
            config = self.base_config.copy()
            config.update(var["config"])
            
            runtime = UnifiedCognitiveRuntime(config)
            runtime.initialize_runtime()
            
            health_scores = []
            for step in range(steps):
                # Simulate a stressful scenario
                noise = torch.randn(1, 1, config["hidden_dim"]) * (step / 50)
                current = [torch.randn(1, 1, config["hidden_dim"]) + noise for _ in range(config["layers"])]
                target = [torch.randn(1, 1, config["hidden_dim"]) for _ in range(config["layers"])]
                
                res = runtime.process_step(current, [], target_hidden=target)
                health_scores.append(res["health"].cognitive_health_score)
            
            summary = runtime.runtime_summary()
            summary["variant"] = var["name"]
            summary["mean_health"] = np.mean(health_scores)
            summary["final_health"] = health_scores[-1]
            results.append(summary)
            
        return results

    def save_results(self, results: List, path: str = "results/phase21/runtime_ablation.json"):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    config = {
        "hidden_dim": 768,
        "layers": 12,
        "max_anchors": 128
    }
    study = RuntimeAblation(config)
    res = study.run_ablation_study()
    study.save_results(res)
    print("Ablation study complete.")

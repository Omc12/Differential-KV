"""
analysis/cognitive_plasticity.py
Phase 17: Cognitive Plasticity Analysis
Studies how reasoning manifolds reorganize after compression collapse or perturbation.
Measures circuit reconfiguration and attractor migration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
import os
import json

class PlasticityAnalyzer(ReasoningTrajectoryTracker):
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        super().__init__(model_id, device)

    def analyze_plasticity(self, prompt: str, noise_levels: List[float]):
        """
        Runs generation at different noise levels and measures how the latent manifold 'migrates'.
        """
        results = []
        for noise in noise_levels:
            print(f"Analyzing plasticity at noise level: {noise}...")
            
            def noise_mod(l, k, v):
                return k + torch.randn_like(k) * noise, v + torch.randn_like(v) * noise
                
            ids, trajectory = self.run_generation(prompt, kv_modifier_fn=noise_mod)
            
            # Measure manifold center
            all_hidden = torch.cat([t["hidden"][-1][:, -1, :].cpu().float() for t in trajectory], dim=0)
            manifold_center = all_hidden.mean(dim=0)
            manifold_variance = all_hidden.var(dim=0).mean().item()
            
            results.append({
                "noise": noise,
                "manifold_center_norm": torch.norm(manifold_center).item(),
                "manifold_variance": manifold_variance,
                "text": self.tokenizer.decode(ids[0])
            })
            
        return results

    def plot_plasticity(self, results, save_path):
        noises = [r["noise"] for r in results]
        variances = [r["manifold_variance"] for r in results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(noises, variances, 'b-o')
        plt.title("Cognitive Plasticity: Manifold Variance vs Noise")
        plt.xlabel("KV Noise Level")
        plt.ylabel("Latent Manifold Variance")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    analyzer = PlasticityAnalyzer()
    prompt = "Explain the theory of relativity in simple terms."
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    res = analyzer.analyze_plasticity(prompt, noise_levels)
    
    os.makedirs("results/phase17/plots", exist_ok=True)
    analyzer.plot_plasticity(res, "results/phase17/plots/plasticity_variance.png")
    
    with open("results/phase17/data/plasticity_results.json", "w") as f:
        json.dump(res, f, indent=4)

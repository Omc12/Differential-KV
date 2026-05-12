"""
experiments/universal_basin_mapping.py
Phase 18: Evolutionary Manifold Shaping
Compares attractor basins across Qwen, Llama, Gemma, and DeepSeek.
"""

import torch
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
from analysis.attractor_mapper import AttractorMapper
from sklearn.decomposition import PCA
import numpy as np

class UniversalBasinMapper:
    def __init__(self, models: List[str], device="cuda"):
        self.models = models
        self.device = device
        self.trackers = {}
        self.mappers = {m: AttractorMapper() for m in models}

    def collect_basins(self, prompt: str):
        """Runs generation on all models and records states."""
        for m_id in self.models:
            print(f"Mapping basins for {m_id}...")
            try:
                tracker = ReasoningTrajectoryTracker(model_id=m_id, device=self.device)
                _, traj = tracker.run_generation(prompt, max_new_tokens=40)
                
                for i, step in enumerate(traj):
                    h = step["hidden"][-1][0, -1, :].numpy()
                    v = h - traj[i-1]["hidden"][-1][0, -1, :].numpy() if i > 0 else np.zeros_like(h)
                    self.mappers[m_id].record_state(h, v, 0.8)
                    
                del tracker
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to map basin for {m_id}: {e}")

    def plot_universal_map(self, save_path: str):
        """Plots all model basins in a shared latent comparison space."""
        plt.figure(figsize=(12, 10))
        
        all_pts = []
        model_indices = []
        valid_models = []
        for m_id in self.models:
            if self.mappers[m_id].points:
                all_pts.extend(self.mappers[m_id].points)
                model_indices.extend([m_id] * len(self.mappers[m_id].points))
                valid_models.append(m_id)
            
        if not all_pts: return
        
        pts = np.array(all_pts)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(pts)
        
        cmap = plt.cm.get_cmap('tab10')
        
        start_idx = 0
        for i, m_id in enumerate(valid_models):
            end_idx = start_idx + len(self.mappers[m_id].points)
            m_coords = coords[start_idx:end_idx]
            plt.scatter(m_coords[:, 0], m_coords[:, 1], label=m_id, color=cmap(i), alpha=0.5)
            plt.plot(m_coords[:, 0], m_coords[:, 1], color=cmap(i), alpha=0.3)
            start_idx = end_idx
            
        plt.title("Universal Cognitive Basin Comparison")
        plt.xlabel("Shared PCA 1")
        plt.ylabel("Shared PCA 2")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Small models to avoid OOM
    models = ["Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"] 
    
    mapper = UniversalBasinMapper(models)
    prompt = "If Alice has 3 apples and Bob gives her 2 more, but then Charlie takes half of them, how many does she have?"
    
    mapper.collect_basins(prompt)
    os.makedirs("results/phase18/plots", exist_ok=True)
    mapper.plot_universal_map("results/phase18/plots/universal_basin_map.png")

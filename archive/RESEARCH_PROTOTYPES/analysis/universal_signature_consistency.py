"""
analysis/universal_signature_consistency.py
Phase 19.5: Universal Validation & Reproducibility Consolidation
Measures the consistency of cognitive collapse signatures across different architectures and scales.
"""

import torch
import numpy as np
import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.universal_collapse_signatures import CollapseSignatureAnalyzer

class SignatureConsistencyAnalyzer:
    def __init__(self, output_dir: str = "results/phase19_5/signatures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.analyzer = CollapseSignatureAnalyzer()
        self.data_store = {} # model_name -> signature_profile

    def add_model_data(self, model_name: str, traj_data: Dict[str, Any]):
        """Records signature profile for a model."""
        profile = self.analyzer.build_collapse_profile(traj_data)
        # Add more advanced metrics for 19.5
        profile["manifold_bifurcation"] = self.compute_bifurcation_index(traj_data["traj"])
        profile["topology_fragmentation"] = self.compute_topology_fragmentation(traj_data["traj"])
        
        self.data_store[model_name] = profile

    def compute_bifurcation_index(self, traj: List[Dict]):
        """Measures how much the trajectory splits into divergent attractors."""
        states = np.array([t["hidden"][-1][0, -1, :].numpy() for t in traj])
        if len(states) < 4: return 0.0
        # Simple heuristic: variance of velocities in the latter half
        velocities = np.diff(states, axis=0)
        latter_half = velocities[len(velocities)//2:]
        return np.var(latter_half).item()

    def compute_topology_fragmentation(self, traj: List[Dict]):
        """Measures attention sparsity/fragmentation trends."""
        # Simplified: mean of entropy diffusion
        attn_weights = [t["attn"][-1][0].cpu() for t in traj]
        entropy = self.analyzer.analyze_entropy_fragmentation(attn_weights)
        return np.mean(entropy).item()

    def generate_universality_report(self):
        """Generates correlation matrices and heatmaps."""
        models = list(self.data_store.keys())
        if not models: return
        
        metrics = ["acceleration", "curvature", "entropy"]
        consistency_matrix = np.zeros((len(models), len(models)))
        
        # Calculate cross-model correlation for acceleration profiles
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                p1 = np.array(self.data_store[m1]["acceleration"])
                p2 = np.array(self.data_store[m2]["acceleration"])
                # Trim to same length
                min_len = min(len(p1), len(p2))
                if min_len > 1:
                    corr = np.corrcoef(p1[:min_len], p2[:min_len])[0, 1]
                    consistency_matrix[i, j] = corr if not np.isnan(corr) else 0.0

        self.plot_heatmap(consistency_matrix, models, "Signature Correlation Matrix (Acceleration)")
        
        # Save aggregate statistics
        stats = {
            model: {
                "avg_accel": np.mean(self.data_store[model]["acceleration"]),
                "max_curvature": np.max(self.data_store[model]["curvature"]),
                "bifurcation": self.data_store[model]["manifold_bifurcation"]
            } for model in models
        }
        
        with open(os.path.join(self.output_dir, "signature_consistency.json"), "w") as f:
            json.dump(stats, f, indent=4)

    def plot_heatmap(self, data, labels, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, xticklabels=labels, yticklabels=labels, cmap="magma")
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

if __name__ == "__main__":
    # Test with dummy data
    sca = SignatureConsistencyAnalyzer()
    # Logic for manual test if needed

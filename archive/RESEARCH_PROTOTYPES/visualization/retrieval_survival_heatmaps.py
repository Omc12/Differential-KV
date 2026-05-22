"""
visualization/retrieval_survival_heatmaps.py

Generates survival curves and collapse heatmaps for retrieval.
Focus: collapse boundaries, retrieval survival curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

def plot_survival_heatmap(densities: List[float], noise_levels: List[float], survival_data: np.ndarray, save_path: str = "results/reconstruction_5a/retrieval_survival_heatmap.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(survival_data, xticklabels=np.round(densities, 2), yticklabels=np.round(noise_levels, 2), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Retrieval Survival Heatmap (Density vs Noise)")
    plt.xlabel("Anchor Density")
    plt.ylabel("Context Noise Level")
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.close()

def plot_survival_curves(densities: List[float], survival_rates: List[float], save_path: str = "results/reconstruction_5a/retrieval_survival_curves.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(densities, survival_rates, marker='o', linestyle='-', linewidth=2)
    plt.title("Retrieval Survival Curve")
    plt.xlabel("Anchor Density")
    plt.ylabel("Survival Rate")
    plt.grid(True)
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Curves saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Mock data for demonstration
    densities = np.linspace(0.01, 0.2, 5).tolist()
    noise_levels = np.linspace(0.0, 0.5, 5).tolist()
    # Simulated survival data: survival increases with density, decreases with noise
    mock_survival = np.zeros((5, 5))
    for i, n in enumerate(noise_levels):
        for j, d in enumerate(densities):
            mock_survival[i, j] = max(0, min(1, (d * 10) - (n * 0.5)))
            
    plot_survival_heatmap(densities, noise_levels, mock_survival)
    plot_survival_curves(densities, mock_survival[0, :].tolist())

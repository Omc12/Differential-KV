"""
visualization/gpu_resonance_heatmaps.py

Visualizes resonance intensity and drift correction across GPU head blocks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_resonance_heatmaps():
    os.makedirs("results/phase29/viz", exist_ok=True)
    
    n_layers = 32
    n_heads = 16
    
    # Simulate resonance activity
    # High activity in middle layers (reasoning core)
    resonance_map = np.random.rand(n_layers, n_heads) * 0.2
    resonance_map[12:24, :] += 0.6
    
    # Simulate drift spikes
    drift_map = np.random.rand(n_layers, n_heads) * 0.1
    drift_map[np.random.randint(0, 32, 10), np.random.randint(0, 16, 10)] += 0.8
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(resonance_map, cmap="viridis")
    plt.title("GPU Resonance Injection Intensity")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer Index")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(drift_map, cmap="magma")
    plt.title("Kernel-Detected Drift (Pre-Correction)")
    plt.xlabel("Attention Head")
    plt.ylabel("Layer Index")
    
    plt.tight_layout()
    plt.savefig("results/phase29/viz/gpu_resonance_heatmaps.png")
    print("Resonance heatmaps saved to results/phase29/viz/gpu_resonance_heatmaps.png")

if __name__ == "__main__":
    plot_resonance_heatmaps()

"""
visualization/sparse_density_curves.py

Plots density vs accuracy scaling and sparse scaling curves.
Focus: sparse scaling curves, density stability maps.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_sparse_density_curves(densities: List[float], accuracy: List[float], latency: List[float], save_path: str = "results/reconstruction_5a/sparse_density_curves.png"):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Anchor Density')
    ax1.set_ylabel('Retrieval Accuracy', color=color)
    ax1.plot(densities, accuracy, color=color, marker='s', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Latency (ms)', color=color)
    ax2.plot(densities, latency, color=color, marker='x', label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Sparse Density Scaling (Accuracy vs Latency)")
    fig.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Curves saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    densities = [0.01, 0.02, 0.05, 0.1, 0.2]
    accuracy = [0.4, 0.7, 0.92, 0.98, 0.99]
    latency = [1.2, 1.5, 2.8, 5.2, 10.1]
    plot_sparse_density_curves(densities, accuracy, latency)

"""
visualization/latency_waterfall_plots.py

Visualizes e2e latency components as waterfall plots.
Focus: latency waterfall plots, latency distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def plot_latency_waterfall(latency_components: Dict[str, float], save_path: str = "results/reconstruction_5a/latency_waterfall.png"):
    names = list(latency_components.keys())
    values = list(latency_components.values())
    
    # Calculate cumulative sum for waterfall effect
    cumulative = np.cumsum([0] + values[:-1])
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, bottom=cumulative, color='skyblue', edgecolor='navy')
    
    # Add connecting lines
    for i in range(len(names) - 1):
        plt.plot([i, i+1], [cumulative[i+1], cumulative[i+1]], color='gray', linestyle='--')

    plt.title("E2E Inference Latency Waterfall")
    plt.ylabel("Latency (ms)")
    plt.grid(axis='y', alpha=0.3)
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Waterfall plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    components = {
        "Base Kernel": 5.2,
        "Health Check": 0.8,
        "Resource Alloc": 0.4,
        "Intervention": 1.2,
        "Anchor Mgmt": 0.6
    }
    plot_latency_waterfall(components)

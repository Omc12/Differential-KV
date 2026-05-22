"""
visualization/kernel_fusion_heatmaps.py

Visualizes kernel fusion efficiency and occupancy heatmaps.
Highlights hotspots and warp divergence reduction.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_kernel_fusion_heatmap(fusion_data: np.ndarray, title: str = "Kernel Fusion Efficiency"):
    """
    Plots a heatmap showing where kernel fusion is most effective.
    X-axis: Token Blocks, Y-axis: Feature Blocks.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(fusion_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Fusion Efficiency (%)')
    plt.title(title)
    plt.xlabel('Token Blocks')
    plt.ylabel('Feature Blocks')
    plt.tight_layout()
    plt.savefig('results/plots/kernel_fusion_heatmap.png')
    plt.close()

if __name__ == "__main__":
    # Mock data for demonstration
    mock_fusion = np.random.rand(16, 16) * 100
    plot_kernel_fusion_heatmap(mock_fusion)

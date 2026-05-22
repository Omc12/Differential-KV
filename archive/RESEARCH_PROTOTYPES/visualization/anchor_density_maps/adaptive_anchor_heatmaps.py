import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_adaptive_anchor_heatmap(density_map: torch.Tensor, anchor_indices: torch.Tensor, save_path: str):
    """
    Visualizes the density map and where the adaptive anchors were placed.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot density map
    density_np = density_map.cpu().numpy()
    plt.plot(density_np, label="Retrieval Density", color="cyan", alpha=0.6)
    
    # Plot anchors as vertical lines or dots
    anchor_np = anchor_indices.cpu().numpy()
    plt.scatter(anchor_np, np.ones_like(anchor_np), color="red", s=10, label="Adaptive Anchors", marker="|")
    
    plt.title("Adaptive Anchor Heatmap & Retrieval Density")
    plt.xlabel("Sequence Position")
    plt.ylabel("Density / Anchor Presence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Mock data for demonstration
    seq_len = 8192
    density = torch.ones(seq_len)
    density[2000:3000] = 0.4 # Simulate hotspot
    anchors = torch.cat([torch.arange(0, seq_len, 512), torch.arange(2000, 3000, 64)])
    plot_adaptive_anchor_heatmap(density, anchors, "results/phase7/adaptive_anchor_heatmap.png")

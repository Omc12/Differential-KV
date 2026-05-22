"""
visualization/attention_basin_maps.py

Visualizes attractor basins and token navigation in the latent manifold.
"""

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_attractor_basins(
    tokens: torch.Tensor,       # [S, D]
    attractors: torch.Tensor,    # [A, D]
    output_path: str = "attractor_basin_map.png"
):
    print(f"Generating Attractor Basin Map: {output_path}")
    
    # 1. Project to 2D
    pca = PCA(n_components=2)
    all_points = torch.cat([tokens, attractors], dim=0).detach().cpu().numpy()
    projected = pca.fit_transform(all_points)
    
    s_len = tokens.shape[0]
    tokens_2d = projected[:s_len]
    attractors_2d = projected[s_len:]
    
    # 2. Create Plot
    plt.figure(figsize=(10, 8))
    
    # Plot basins (density-like)
    for i in range(len(attractors_2d)):
        circle = plt.Circle(attractors_2d[i], 0.5, color='blue', alpha=0.1)
        plt.gca().add_patch(circle)
        plt.scatter(attractors_2d[i, 0], attractors_2d[i, 1], c='red', marker='x', label='Attractor' if i==0 else "")
        
    # Plot tokens
    plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c='green', alpha=0.5, s=10, label='Tokens')
    
    # Plot trajectories (lines between consecutive tokens)
    plt.plot(tokens_2d[:, 0], tokens_2d[:, 1], c='green', alpha=0.2)
    
    plt.title("NCAA Attractor Basin Navigation Map")
    plt.xlabel("Geometric Principal Component 1")
    plt.ylabel("Geometric Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    S, D, A = 512, 64, 4
    tokens = torch.randn(S, D)
    attractors = torch.randn(A, D)
    
    plot_attractor_basins(tokens, attractors)

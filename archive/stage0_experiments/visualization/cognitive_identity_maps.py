import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
from typing import List

def plot_cognitive_identity_map(manifolds: torch.Tensor, labels: List[str], save_path: str = "identity_map.png"):
    """
    Generates a 2D map of cognitive identity manifolds using t-SNE.
    """
    print(f"Generating cognitive identity map to {save_path}...")
    
    # manifolds: [num_identities, n, d]
    num_ids, n, d = manifolds.shape
    reshaped = manifolds.view(-1, d).detach().cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=min(30, n*num_ids-1))
    reduced = tsne.fit_transform(reshaped)
    
    plt.figure(figsize=(10, 8))
    for i in range(num_ids):
        start = i * n
        end = (i + 1) * n
        plt.scatter(reduced[start:end, 0], reduced[start:end, 1], label=labels[i], alpha=0.6)
        
    plt.title("Persistent Cognitive Identity Map")
    plt.xlabel("Manifold Dimension 1")
    plt.ylabel("Manifold Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    # Demo
    m1 = torch.randn(1, 100, 64) + 2.0
    m2 = torch.randn(1, 100, 64) - 2.0
    plot_cognitive_identity_map(torch.cat([m1, m2], dim=0), ["Agent Alpha", "Agent Beta"])

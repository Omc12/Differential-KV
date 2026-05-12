"""
visualization/cognitive_head_roles.py

Visualizes the allocation of specialized roles across attention heads.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_head_role_distribution(
    role_allocations: dict, # {role: [H]}
    output_path: str = "head_role_distribution.png"
):
    print(f"Generating Head Role Distribution: {output_path}")
    
    roles = list(role_allocations.keys())
    n_heads = len(role_allocations[roles[0]])
    
    # Prepare data for stacked bar chart
    data = []
    for role in roles:
        data.append(role_allocations[role])
    data = np.array(data)
    
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(n_heads)
    
    for i, role in enumerate(roles):
        plt.bar(range(n_heads), data[i], bottom=bottom, label=role)
        bottom += data[i]
        
    plt.title("NCAA Cognitive Head Role Allocation")
    plt.xlabel("Attention Head Index")
    plt.ylabel("Role Allocation Probability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    H = 32
    roles = ["retrieval", "stabilization", "predictive", "resonance", "routing"]
    mock_allocs = {}
    for r in roles:
        mock_allocs[r] = np.random.dirichlet([1]*5, size=H).T[roles.index(r)]
        
    plot_head_role_distribution(mock_allocs)

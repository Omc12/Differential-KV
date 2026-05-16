"""
visualization/cognition_scaling_maps.py

Generates 3D visualizations of cognitive scaling and manifold stability.
Maps context length, sparsity ratio, and reasoning retention into a unified manifold.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class CognitionScalingMapper:
    def __init__(self):
        self.output_dir = "results/phase38/plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_3d_map(self):
        print("Generating 3D Cognition Scaling Map...")
        
        # Simulated data
        ctx_len = np.linspace(32, 512, 20) # k-tokens
        sparsity = np.linspace(0.01, 0.5, 20)
        X, Y = np.meshgrid(ctx_len, sparsity)
        
        # Stability metric (Z)
        # Higher is more stable. Stability drops as context increases and sparsity decreases too much.
        Z = np.exp(-X/512) * (1 - np.abs(Y - 0.1))
        
        fig = plt.figure(figsize=(12, 8), facecolor='#0f172a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0f172a')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel('Context Length (k-tokens)', color='#94a3b8')
        ax.set_ylabel('Sparsity Ratio', color='#94a3b8')
        ax.set_zlabel('Stability Score', color='#94a3b8')
        ax.set_title('Cognitive Scaling Manifold', color='white', fontsize=16)
        
        # Customize ticks
        ax.tick_params(axis='x', colors='#64748b')
        ax.tick_params(axis='y', colors='#64748b')
        ax.tick_params(axis='z', colors='#64748b')
        
        plt.savefig(os.path.join(self.output_dir, "cognition_scaling_3d.png"), dpi=300)
        print(f"3D Map saved to {self.output_dir}")

if __name__ == "__main__":
    mapper = CognitionScalingMapper()
    mapper.generate_3d_map()

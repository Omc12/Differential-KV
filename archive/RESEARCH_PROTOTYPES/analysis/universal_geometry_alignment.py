"""
analysis/universal_geometry_alignment.py
Phase 19: Universal Cognitive Geometry
Aligns latent manifolds across diverse model architectures (Qwen, Llama, Gemma, Mistral, etc.).
"""

import torch
import numpy as np
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class LatentManifoldAligner:
    def __init__(self):
        self.alignments = {} # (source, target) -> transformation

    def procrustes_align(self, source_states: np.ndarray, target_states: np.ndarray):
        """
        Aligns source_states to target_states using Procrustes analysis.
        Returns: mtx1, mtx2, disparity
        """
        # States must have same shape for scipy procrustes
        # If dimensions differ, we use PCA to reduce to common dim or pad
        d1 = source_states.shape[1]
        d2 = target_states.shape[1]
        
        if d1 != d2:
            common_dim = min(d1, d2, 128)
            pca1 = PCA(n_components=common_dim)
            pca2 = PCA(n_components=common_dim)
            s1 = pca1.fit_transform(source_states)
            s2 = pca2.fit_transform(target_states)
        else:
            s1 = source_states
            s2 = target_states

        mtx1, mtx2, disparity = procrustes(s2, s1)
        return mtx1, mtx2, disparity

    def compute_cca_alignment(self, source_states: np.ndarray, target_states: np.ndarray, n_components: int = 10):
        """
        Aligns using Canonical Correlation Analysis.
        """
        cca = CCA(n_components=n_components)
        s1_c, s2_c = cca.fit_transform(source_states, target_states)
        correlation = np.corrcoef(s1_c.T, s2_c.T).diagonal(offset=n_components)
        return s1_c, s2_c, correlation

    def visualize_alignment(self, mtx1: np.ndarray, mtx2: np.ndarray, source_name: str, target_name: str, save_path: str):
        plt.figure(figsize=(10, 8))
        plt.scatter(mtx1[:, 0], mtx1[:, 1], label=target_name, alpha=0.6, c='blue')
        plt.scatter(mtx2[:, 0], mtx2[:, 1], label=f"{source_name} (Aligned)", alpha=0.6, c='red')
        
        # Draw lines between corresponding points
        for i in range(len(mtx1)):
            plt.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 1], mtx2[i, 1]], 'k-', alpha=0.1)
            
        plt.title(f"Universal Geometry Alignment: {source_name} -> {target_name}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Mock test
    aligner = LatentManifoldAligner()
    s1 = np.random.randn(100, 768)
    s2 = s1 @ np.random.randn(768, 1024) + np.random.randn(1024) # Affine transform + noise
    s2 = s2[:, :1024] # Ensure shape
    
    # Pad or truncate for procrustes mock
    s1_p = s1
    s2_p = s2[:, :768]
    
    mtx1, mtx2, disp = aligner.procrustes_align(s1_p, s2_p)
    print(f"Procrustes Disparity: {disp}")
    
    import os
    os.makedirs("results/phase19/plots", exist_ok=True)
    aligner.visualize_alignment(mtx1, mtx2, "SourceModel", "TargetModel", "results/phase19/plots/geometry_alignment_mock.png")

"""
analysis/stability_basin_clustering.py
Phase 26: Cognitive Energy Minimization (CEM)
Clusters low-energy reasoning trajectories to identify self-sustaining states.
"""

import numpy as np
from typing import List, Dict, Optional
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    # Fallback if scikit-learn is not available during immediate execution
    DBSCAN = None

class StabilityBasinClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.data_points = []
        self.energies = []
        self.labels = None

    def add_trajectory_point(self, latent_state: np.ndarray, energy: float):
        """Records a point if it's considered stable (low energy)."""
        if energy < 0.2: # Energy threshold for "stable enough to cluster"
            self.data_points.append(latent_state.flatten())
            self.energies.append(energy)

    def cluster_basins(self) -> int:
        """Clusters the recorded points into stability basins."""
        if len(self.data_points) < self.min_samples or DBSCAN is None:
            return 0
            
        X = np.array(self.data_points)
        # Normalize X to help DBSCAN
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_norm)
        self.labels = clustering.labels_
        
        num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return num_clusters

    def get_centroids(self) -> List[np.ndarray]:
        """Returns the centroids of the discovered stability basins."""
        if self.labels is None or len(self.data_points) == 0:
            return []
            
        X = np.array(self.data_points)
        centroids = []
        unique_labels = set(self.labels)
        for label in unique_labels:
            if label == -1: continue # Skip noise
            centroids.append(np.mean(X[self.labels == label], axis=0))
        return centroids

    def get_basin_survival_scores(self) -> Dict[int, float]:
        """Ranks basins by their average energy (lower is better survival)."""
        if self.labels is None:
            return {}
            
        scores = {}
        unique_labels = set(self.labels)
        for label in unique_labels:
            if label == -1: continue
            basin_energies = [self.energies[i] for i, l in enumerate(self.labels) if l == label]
            scores[int(label)] = float(1.0 - np.mean(basin_energies)) # Higher score = better
        return scores

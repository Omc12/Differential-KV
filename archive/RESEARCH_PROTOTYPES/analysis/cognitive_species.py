"""
analysis/cognitive_species.py
Phase 18: Evolutionary Manifold Shaping
Clusters evolved manifold types to identify distinct reasoning geometries.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class CognitiveSpeciesAnalyzer:
    def __init__(self, n_species: int = 4):
        self.n_species = n_species
        self.kmeans = KMeans(n_clusters=n_species, n_init=10)
        self.pca = PCA(n_components=2)
        self.species_centroids = None

    def cluster_manifolds(self, population_states: List[torch.Tensor]) -> np.ndarray:
        """
        Clusters a population of manifolds into 'cognitive species'.
        population_states: List of flattened mean hidden states or manifold descriptors.
        """
        data = torch.stack(population_states).detach().cpu().numpy()
        labels = self.kmeans.fit_predict(data)
        self.species_centroids = self.kmeans.cluster_centers_
        return labels

    def visualize_species(self, population_states: List[torch.Tensor], labels: np.ndarray, save_path: str):
        """
        Visualizes the 'phylogenetic' tree of cognitive species.
        """
        data = torch.stack(population_states).detach().cpu().numpy()
        coords = self.pca.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Species ID')
        
        # Plot centroids
        centroid_coords = self.pca.transform(self.species_centroids)
        plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], c='red', marker='X', s=200, label='Centroids')
        
        plt.title("Clustered Cognitive Species (Reasoning Geometries)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def identify_species_traits(self, labels: np.ndarray, population_metrics: List[Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        """
        Summarizes the traits of each cognitive species.
        """
        species_traits = {}
        for species_id in range(self.n_species):
            indices = np.where(labels == species_id)[0]
            if len(indices) == 0: continue
            
            traits = {
                "avg_fitness": np.mean([population_metrics[i]["unified_fitness"] for i in indices]),
                "avg_continuity": np.mean([population_metrics[i]["continuity"] for i in indices]),
                "avg_recovery": np.mean([population_metrics[i]["recovery_prob"] for i in indices]),
                "population_share": len(indices) / len(labels)
            }
            species_traits[species_id] = traits
            
        return species_traits

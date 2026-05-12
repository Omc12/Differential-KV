"""
analysis/geometry_invariant_reasoning.py
Phase 19: Universal Cognitive Geometry
Searches for invariant latent structures across different reasoning tasks.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

class ReasoningGeometryInvariance:
    def __init__(self):
        self.task_geometries = {}

    def record_task_geometry(self, task_name: str, trajectory: List[np.ndarray]):
        states = np.array(trajectory)
        pca = PCA(n_components=10)
        transformed = pca.fit_transform(states)
        explained_var = pca.explained_variance_ratio_
        
        self.task_geometries[task_name] = {
            "coords": transformed,
            "variance": explained_var,
            "centroid": np.mean(states, axis=0)
        }

    def compare_invariants(self):
        """
        Compares centroids and variance distributions across tasks.
        """
        tasks = list(self.task_geometries.keys())
        if len(tasks) < 2: return {}
        
        comparison = {}
        for i in range(len(tasks)):
            for j in range(i+1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                c1 = self.task_geometries[t1]["centroid"]
                c2 = self.task_geometries[t2]["centroid"]
                dist = np.linalg.norm(c1 - c2)
                
                v1 = self.task_geometries[t1]["variance"]
                v2 = self.task_geometries[t2]["variance"]
                v_sim = np.corrcoef(v1, v2)[0, 1]
                
                comparison[f"{t1}_vs_{t2}"] = {
                    "centroid_dist": float(dist),
                    "variance_correlation": float(v_sim)
                }
        return comparison

    def plot_task_comparison(self, save_path: str):
        plt.figure(figsize=(10, 6))
        for task, data in self.task_geometries.items():
            coords = data["coords"]
            plt.scatter(coords[:, 0], coords[:, 1], label=task, alpha=0.5)
            plt.plot(coords[:, 0], coords[:, 1], alpha=0.3)
            
        plt.title("Geometry-Invariant Reasoning Comparison")
        plt.xlabel("Invariant PCA 1")
        plt.ylabel("Invariant PCA 2")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    rgi = ReasoningGeometryInvariance()
    rgi.record_task_geometry("Arithmetic", [np.random.randn(768) for _ in range(10)])
    rgi.record_task_geometry("Coding", [np.random.randn(768) for _ in range(10)])
    
    comp = rgi.compare_invariants()
    print("Task Comparison:", comp)

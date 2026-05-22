"""
analysis/internal_anchor_analysis.py
Phase 17: Self-Organizing Cognitive Memory (SOCM)
Investigates whether models spontaneously develop persistent latent control points
and reusable semantic coordinates under reasoning pressure.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
import os
import json

class InternalAnchorAnalyzer(ReasoningTrajectoryTracker):
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        super().__init__(model_id, device)
        self.latent_history = [] # List of hidden states from multiple runs
        self.metadata = [] # List of (token_id, step, task_id)

    def collect_states(self, prompts: List[str], max_new_tokens=50):
        print(f"Collecting latent states for {len(prompts)} tasks...")
        for task_id, prompt in enumerate(prompts):
            _, trajectory = self.run_generation(prompt, max_new_tokens=max_new_tokens)
            for step, step_data in enumerate(trajectory):
                # We focus on the last layer's hidden state as a representative of the "concept"
                last_hidden = step_data["hidden"][-1][:, -1, :].cpu().float()
                self.latent_history.append(last_hidden)
                self.metadata.append({
                    "task_id": task_id,
                    "step": step,
                    "token": step_data["token"],
                    "text": self.tokenizer.decode([step_data["token"]])
                })

    def analyze_emergent_anchors(self, eps=0.5, min_samples=3):
        """
        Uses DBSCAN to find clusters in latent space across tasks.
        Clusters represent "reusable semantic coordinates" or "attractors".
        """
        if not self.latent_history:
            return {}

        states = torch.cat(self.latent_history, dim=0).numpy()
        print(f"Clustering {len(states)} latent states...")
        
        # Normalize states for better clustering
        states_norm = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-9)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(states_norm)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        print(f"Found {n_clusters} emergent anchor clusters.")
        
        # Analyze clusters
        clusters = {}
        for label in unique_labels:
            if label == -1: continue # Noise
            
            indices = np.where(labels == label)[0]
            cluster_metadata = [self.metadata[i] for i in indices]
            
            # Check for cross-task reuse
            task_ids = set([m["task_id"] for m in cluster_metadata])
            reuse_count = len(task_ids)
            
            # Common tokens in this cluster
            tokens = [m["text"].strip() for m in cluster_metadata]
            from collections import Counter
            common_tokens = Counter(tokens).most_common(5)
            
            clusters[int(label)] = {
                "size": len(indices),
                "reuse_count": reuse_count,
                "common_tokens": common_tokens,
                "representative_state": np.mean(states[indices], axis=0).tolist()
            }
            
        return clusters

    def plot_anchors(self, clusters, save_path):
        if not self.latent_history: return
        
        states = torch.cat(self.latent_history, dim=0).numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(states)
        
        plt.figure(figsize=(10, 8))
        
        # Plot all points in light gray
        plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', alpha=0.3, s=20, label='Latent States')
        
        # Extract cluster labels from metadata if we have them, or re-cluster for plotting
        # Let's use the cluster results we have
        colors = plt.cm.get_cmap('tab10', len(clusters))
        
        for i, (label, data) in enumerate(clusters.items()):
            # Plot the representative state (mean)
            rep = np.array(data["representative_state"]).reshape(1, -1)
            rep_coord = pca.transform(rep)
            plt.scatter(rep_coord[0, 0], rep_coord[0, 1], color=colors(i), s=200, marker='*', edgecolors='black', label=f"Cluster {label}")
            
        plt.title("Phase 17: Emergent Latent Anchors")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    analyzer = InternalAnchorAnalyzer()
    
    prompts = [
        "Question: If x + 5 = 10, what is x? Answer:",
        "Question: If y - 3 = 7, what is y? Answer:",
        "Question: What is 2 + 2? Answer:",
        "Translate 'Hello' to French: ",
        "Translate 'Goodbye' to French: ",
        "Code a python function to add two numbers: ",
        "Code a python function to multiply two numbers: "
    ]
    
    analyzer.collect_states(prompts)
    clusters = analyzer.analyze_emergent_anchors(eps=0.1)
    
    os.makedirs("results/phase17/data", exist_ok=True)
    os.makedirs("results/phase17/plots", exist_ok=True)
    
    analyzer.plot_anchors(clusters, "results/phase17/plots/internal_anchors.png")
    
    print("\nTop 5 Emergent Clusters:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['size'], reverse=True)[:5]
    for label, data in sorted_clusters:
        print(f"Cluster {label}: Size {data['size']}, Reuse {data['reuse_count']}, Common: {data['common_tokens']}")

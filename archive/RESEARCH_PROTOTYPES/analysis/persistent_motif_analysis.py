"""
analysis/persistent_motif_analysis.py
Phase 17: Persistent Motif Analysis
Searches for reusable reasoning motifs and persistent latent circuits in transformer cognition.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
import os
import json

class PersistentMotifAnalyzer(ReasoningTrajectoryTracker):
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        super().__init__(model_id, device)
        self.motif_library = []

    def extract_attention_motifs(self, prompt: str, window_size: int = 5):
        """
        Extracts sequences of attention patterns (motifs) from a single run.
        """
        _, trajectory = self.run_generation(prompt, max_new_tokens=40)
        
        # We focus on a specific layer's attention for motif search
        # Or an aggregate across layers.
        # Motif = sequence of [heads, q_len, k_len]
        
        motifs = []
        for i in range(len(trajectory) - window_size):
            window = [trajectory[j]["attn"][-1][0].cpu() for j in range(i, i + window_size)]
            # Flatten or summarize the window
            # For simplicity, let's take the mean attention distribution of the last query
            summary = torch.stack([w[:, -1, :] for w in window]).mean(dim=0)
            motifs.append(summary)
            
        return motifs

    def find_persistent_motifs(self, prompts: List[str]):
        """
        Compares motifs across different prompts to find persistent ones.
        """
        all_motifs = []
        for prompt in prompts:
            all_motifs.extend(self.extract_attention_motifs(prompt))
            
        print(f"Extracted {len(all_motifs)} potential motifs.")
        
        # Use clustering to find "common" motifs
        if not all_motifs: return []
        
        motifs_flat = torch.stack(all_motifs).view(len(all_motifs), -1).numpy()
        
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(10, len(motifs_flat))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(motifs_flat)
        
        # Representative motifs
        centers = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)
        
        persistent_motifs = []
        for i in range(n_clusters):
            persistent_motifs.append({
                "cluster_id": i,
                "persistence_count": int(counts[i]),
                "motif_vector": centers[i].tolist()
            })
            
        return persistent_motifs

if __name__ == "__main__":
    analyzer = PersistentMotifAnalyzer()
    
    prompts = [
        "If a = 5, b = 6, then a + b =",
        "If x = 10, y = 2, then x * y =",
        "If p = 3, q = 4, then p^2 + q^2 =",
    ]
    
    motifs = analyzer.find_persistent_motifs(prompts)
    
    print(f"\nFound {len(motifs)} persistent motifs.")
    for m in sorted(motifs, key=lambda x: x['persistence_count'], reverse=True)[:3]:
        print(f"Motif {m['cluster_id']}: Persistence {m['persistence_count']}")
        
    os.makedirs("results/phase17/data", exist_ok=True)
    with open("results/phase17/data/persistent_motifs.json", "w") as f:
        json.dump(motifs, f, indent=4)

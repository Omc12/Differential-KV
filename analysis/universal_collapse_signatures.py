"""
analysis/universal_collapse_signatures.py
Phase 19: Universal Cognitive Geometry
Identifies shared collapse precursors across architectures.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

class CollapseSignatureAnalyzer:
    def __init__(self):
        self.signatures = []

    def compute_latent_acceleration(self, trajectory: List[np.ndarray]):
        """
        Computes the second derivative of the latent trajectory.
        High acceleration often precedes collapse.
        """
        states = np.array(trajectory)
        velocities = np.diff(states, axis=0)
        accelerations = np.diff(velocities, axis=0)
        accel_norms = np.linalg.norm(accelerations, axis=-1)
        return accel_norms

    def compute_curvature_spikes(self, trajectory: List[np.ndarray]):
        """
        Measures the curvature of the manifold path.
        """
        states = np.array(trajectory)
        v1 = np.diff(states, axis=0)
        # Normalize velocities
        v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-9)
        # Change in direction
        direction_change = np.linalg.norm(np.diff(v1_norm, axis=0), axis=-1)
        return direction_change

    def analyze_entropy_fragmentation(self, attention_weights: List[torch.Tensor]):
        """
        Measures fragmentation of attention (high entropy across many tokens).
        """
        fragmentation = []
        for attn in attention_weights:
            # attn: [heads, q_len, k_len]
            # Use last query
            q_attn = attn[:, -1, :]
            entropy = -torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1)
            fragmentation.append(entropy.mean().item())
        return fragmentation

    def build_collapse_profile(self, traj_data: Dict[str, Any]):
        """
        Aggregates metrics to find 'The Signature'.
        """
        hidden_states = [t["hidden"][-1][0, -1, :].numpy() for t in traj_data["traj"]]
        attn_weights = [t["attn"][-1][0].cpu() for t in traj_data["traj"]]
        
        accel = self.compute_latent_acceleration(hidden_states)
        curvature = self.compute_curvature_spikes(hidden_states)
        entropy = self.analyze_entropy_fragmentation(attn_weights)
        
        return {
            "acceleration": accel.tolist(),
            "curvature": curvature.tolist(),
            "entropy": entropy
        }

if __name__ == "__main__":
    analyzer = CollapseSignatureAnalyzer()
    # Mock analysis
    mock_traj = [np.random.randn(768) for _ in range(20)]
    accel = analyzer.compute_latent_acceleration(mock_traj)
    print(f"Max Acceleration: {np.max(accel)}")

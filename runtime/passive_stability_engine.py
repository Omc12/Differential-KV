"""
runtime/passive_stability_engine.py
Phase 26: Cognitive Energy Minimization (CEM)
Detects naturally stable latent manifolds to reduce active repair dependency.
"""

import torch
import numpy as np
from typing import List, Dict, Optional

class PassiveStabilityEngine:
    def __init__(self, stability_threshold: float = 0.2):
        self.stability_threshold = stability_threshold
        self.known_stable_manifolds = [] # Centroids of stable basins

    def is_state_stable(self, latent_state: torch.Tensor) -> bool:
        """
        Checks if the current latent state is close to a known stable manifold.
        """
        if not self.known_stable_manifolds:
            return False
            
        # We handle both single vectors and batch/sequence states by flattening
        state_flat = latent_state.detach().cpu().float().numpy().flatten()
        
        # In practice, we might want to use a more efficient search (e.g. KDTree or FAISS)
        # but for this research phase, a simple distance check is sufficient.
        for centroid in self.known_stable_manifolds:
            # Normalize by state dimension to make threshold scale-invariant
            dist = np.linalg.norm(state_flat - centroid) / np.sqrt(state_flat.size)
            if dist < self.stability_threshold:
                return True
        return False

    def register_stable_state(self, latent_state: torch.Tensor):
        """Adds a discovered stable state to the engine's library."""
        state_flat = latent_state.detach().cpu().float().numpy().flatten()
        
        # Avoid redundant entries
        if not self.is_state_stable(latent_state):
            self.known_stable_manifolds.append(state_flat)
            
    def clear(self):
        self.known_stable_manifolds = []

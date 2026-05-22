import torch
import numpy as np
from typing import List, Dict

class ResonancePhaseTracker:
    """
    Tracks the phase of geometric trajectories across layers to detect desynchronization.
    """
    def __init__(self, num_layers: int, history_size: int = 10):
        self.num_layers = num_layers
        self.history_size = history_size
        self.history = [[] for _ in range(num_layers)]
        
    def update(self, layer_idx: int, latent_state: torch.Tensor):
        state = latent_state.detach().cpu().float().numpy().flatten()
        self.history[layer_idx].append(state)
        if len(self.history[layer_idx]) > self.history_size:
            self.history[layer_idx].pop(0)
            
    def compute_phase_lags(self) -> np.ndarray:
        """
        Computes the phase lag between adjacent layers.
        Uses cross-correlation to find the temporal shift that maximizes alignment.
        """
        lags = np.zeros(self.num_layers - 1)
        
        for i in range(self.num_layers - 1):
            h_i = self.history[i]
            h_j = self.history[i+1]
            
            if len(h_i) < 2 or len(h_j) < 2:
                continue
                
            # Compute temporal correlation
            # Simple approach: check dot product of h_i[t] with h_j[t] vs h_j[t-1]
            # More complex: full cross-correlation
            
            # For simplicity in this phase, we use the cosine distance drift
            # as a proxy for phase lag in the latent manifold.
            v_i = h_i[-1]
            v_j = h_j[-1]
            
            v_i_prev = h_i[-2]
            v_j_prev = h_j[-2]
            
            # Velocity vectors
            vel_i = v_i - v_i_prev
            vel_j = v_j - v_j_prev
            
            norm_i = np.linalg.norm(vel_i)
            norm_j = np.linalg.norm(vel_j)
            
            if norm_i > 1e-6 and norm_j > 1e-6:
                cos_sim = np.dot(vel_i, vel_j) / (norm_i * norm_j)
                # Phase lag is inverse of alignment of velocities
                lags[i] = 1.0 - cos_sim
                
        return lags

    def detect_desync(self, threshold: float = 0.5) -> List[int]:
        """Identifies layers that are out of sync with their neighbors."""
        lags = self.compute_phase_lags()
        desync_layers = []
        for i, lag in enumerate(lags):
            if lag > threshold:
                desync_layers.append(i + 1) # Layer j is out of sync with layer i
        return desync_layers

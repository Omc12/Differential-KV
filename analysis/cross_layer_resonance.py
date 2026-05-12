import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ResonanceMetrics:
    alignment_matrix: np.ndarray
    coherence_score: float
    synchronization_entropy: float
    mean_phase_lag: float

class CrossLayerResonanceAnalyzer:
    """
    Analyzes geometric resonance across transformer layers.
    Focuses on inter-layer alignment and phase synchronization.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layer_trajectories = [[] for _ in range(num_layers)]
        
    def add_state(self, layer_idx: int, latent_state: torch.Tensor):
        """Adds a latent state (e.g., hidden states or KV features) for a layer."""
        # Flatten and detach
        flat_state = latent_state.detach().cpu().float().numpy().flatten()
        self.layer_trajectories[layer_idx].append(flat_state)
        
    def compute_resonance(self) -> ResonanceMetrics:
        """Computes resonance metrics across all layers."""
        alignment_matrix = np.zeros((self.num_layers, self.num_layers))
        
        # Convert trajectories to numpy arrays
        trajectories = []
        for t in self.layer_trajectories:
            if len(t) == 0:
                trajectories.append(None)
                continue
            trajectories.append(np.stack(t))
            
        # Compute pairwise cosine alignment
        for i in range(self.num_layers):
            for j in range(i, self.num_layers):
                if trajectories[i] is not None and trajectories[j] is not None:
                    # Use last state for alignment
                    v_i = trajectories[i][-1]
                    v_j = trajectories[j][-1]
                    
                    norm_i = np.linalg.norm(v_i)
                    norm_j = np.linalg.norm(v_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        alignment = np.dot(v_i, v_j) / (norm_i * norm_j)
                        alignment_matrix[i, j] = alignment
                        alignment_matrix[j, i] = alignment
        
        # Resonance coherence (mean of the alignment matrix)
        coherence_score = float(np.mean(alignment_matrix))
        
        # Synchronization Entropy (based on eigenvalues of alignment matrix)
        evals = np.linalg.eigvalsh(alignment_matrix + np.eye(self.num_layers) * 1e-6)
        evals = np.maximum(evals, 1e-10)
        evals /= np.sum(evals)
        sync_entropy = -np.sum(evals * np.log2(evals))
        
        # Mean Phase Lag (dummy for now, will be implemented in phase_tracker)
        mean_phase_lag = 0.0 
        
        return ResonanceMetrics(
            alignment_matrix=alignment_matrix,
            coherence_score=coherence_score,
            synchronization_entropy=sync_entropy,
            mean_phase_lag=mean_phase_lag
        )

    def reset(self):
        self.layer_trajectories = [[] for _ in range(self.num_layers)]

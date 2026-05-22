"""
training/trajectory_feature_extractor.py

Advanced feature extraction from hidden state trajectories.
Extracts statistical, geometric, and topological features for regime classification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

class TrajectoryFeatureExtractor:
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 32):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.history = []
        
    def extract(self, current_hidden: torch.Tensor, prev_hidden: torch.Tensor = None) -> Dict[str, Any]:
        """
        current_hidden: [layers, batch, seq, dim]
        """
        # Focus on the last token of the last layer or mean across layers
        # layers_mean = current_hidden.mean(dim=0) # [batch, seq, dim]
        last_token = current_hidden[:, :, -1, :] # [layers, batch, dim]
        
        # 1. Statistical Features
        mean = last_token.mean().item()
        std = last_token.std().item()
        skew = self._calculate_skew(last_token)
        kurtosis = self._calculate_kurtosis(last_token)
        
        # 2. Geometric Features (Drift, Velocity)
        drift = 0.0
        velocity = 0.0
        curvature = 0.0
        if prev_hidden is not None:
            prev_last = prev_hidden[:, :, -1, :]
            diff = last_token - prev_last
            velocity = torch.norm(diff, p=2).item()
            drift = velocity / (torch.norm(prev_last, p=2).item() + 1e-9)
            
            # Curvature (angle change between consecutive velocities)
            # Requires tracking the previous velocity
            pass 

        # 3. Spectral Features
        # Energy in different frequency components of the hidden state vector
        fft = torch.fft.fft(last_token.float())
        spectral_entropy = -(fft.abs() * torch.log(fft.abs() + 1e-9)).mean().item()
        
        return {
            "mean": mean,
            "std": std,
            "skew": skew,
            "kurtosis": kurtosis,
            "latent_drift": drift,
            "token_velocity": velocity,
            "spectral_entropy": spectral_entropy,
            "layer_variance": last_token.var(dim=0).mean().item()
        }

    def _calculate_skew(self, x: torch.Tensor):
        mu = x.mean()
        sigma = x.std()
        return ((x - mu)**3).mean().item() / (sigma**3 + 1e-9)

    def _calculate_kurtosis(self, x: torch.Tensor):
        mu = x.mean()
        sigma = x.std()
        return ((x - mu)**4).mean().item() / (sigma**4 + 1e-9) - 3.0

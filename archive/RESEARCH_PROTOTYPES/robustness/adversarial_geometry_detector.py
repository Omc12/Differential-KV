import torch
import torch.nn as nn
from typing import Dict, Optional

class AdversarialGeometryDetector:
    """
    Detects adversarial drift and malicious geometry perturbations.
    Identifies attempts to destabilize the reasoning manifold.
    """
    def __init__(self, d_model: int, sensitivity: float = 2.0):
        self.d_model = d_model
        self.sensitivity = sensitivity
        self.drift_threshold = 0.15
        
    def detect_adversarial_drift(self, manifold_states: torch.Tensor, expected_manifold: torch.Tensor) -> Dict:
        """
        Calculates the divergence from the expected manifold.
        """
        # Manifold distance (Euclidean or Cosine)
        dist = torch.norm(manifold_states - expected_manifold, dim=-1).mean()
        
        # Spectral norm of the drift as an indicator of adversarial intent
        drift_vector = manifold_states - expected_manifold
        
        is_adversarial = dist > self.drift_threshold
        
        # Check for high-frequency perturbations (common in adversarial attacks)
        fft_drift = torch.fft.rfft(drift_vector, dim=1)
        hf_energy = torch.norm(fft_drift[:, -10:], dim=-1).mean() # Last 10 frequency bins
        
        if hf_energy > self.sensitivity:
            is_adversarial = True
            
        return {
            "is_adversarial": is_adversarial,
            "drift_magnitude": dist.item(),
            "hf_noise_level": hf_energy.item(),
            "risk_score": (dist.item() / self.drift_threshold) + (hf_energy.item() / self.sensitivity)
        }

    def detect_collapse_trap(self, curvature: torch.Tensor) -> bool:
        """Detects if the manifold is being pushed into a singular region."""
        # High curvature spikes often precede collapse
        return curvature.max() > 10.0

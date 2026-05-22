"""
analysis/basin_fitness.py
Phase 18: Evolutionary Manifold Shaping
Computes fitness metrics for reasoning attractor basins.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional

class BasinFitnessEvaluator:
    def __init__(self, device="cuda"):
        self.device = device

    def compute_fitness(self, 
                        trajectory: List[torch.Tensor], 
                        logits: List[torch.Tensor],
                        retrieval_accuracies: List[float],
                        recovery_events: List[bool]) -> Dict[str, float]:
        """
        Computes a unified fitness score based on multiple cognitive metrics.
        """
        # 1. Reasoning Continuity: smoothness of latent trajectory
        continuity = self._compute_continuity(trajectory)
        
        # 2. Semantic Coherence: entropy stability in logits
        coherence = self._compute_coherence(logits)
        
        # 3. Retrieval Survival: ability to recall facts under compression
        survival = np.mean(retrieval_accuracies) if retrieval_accuracies else 0.0
        
        # 4. Recovery Probability: likelihood of returning to stable manifold after perturbation
        recovery_prob = np.mean(recovery_events) if recovery_events else 0.0
        
        # 5. Latent Entropy: dispersion of hidden states (lower is usually more stable)
        latent_entropy = self._compute_latent_entropy(trajectory)
        
        # 6. Manifold Smoothness: local curvature of the manifold
        smoothness = self._compute_manifold_smoothness(trajectory)
        
        # Unified Score (weighted sum)
        weights = {
            "continuity": 0.2,
            "coherence": 0.2,
            "survival": 0.2,
            "recovery": 0.2,
            "latent_entropy": -0.1, # Penalty for high entropy
            "smoothness": 0.1
        }
        
        unified_score = (
            weights["continuity"] * continuity +
            weights["coherence"] * coherence +
            weights["survival"] * survival +
            weights["recovery"] * recovery_prob +
            weights["latent_entropy"] * latent_entropy +
            weights["smoothness"] * smoothness
        )
        
        return {
            "unified_fitness": float(unified_score),
            "continuity": float(continuity),
            "coherence": float(coherence),
            "survival": float(survival),
            "recovery_prob": float(recovery_prob),
            "latent_entropy": float(latent_entropy),
            "smoothness": float(smoothness)
        }

    def _compute_continuity(self, trajectory: List[torch.Tensor]) -> float:
        """Measures L2 distance between successive states. Lower drift = higher continuity."""
        if len(trajectory) < 2: return 1.0
        drifts = []
        for i in range(len(trajectory) - 1):
            drift = torch.norm(trajectory[i+1] - trajectory[i], p=2).item()
            drifts.append(drift)
        # Normalize: higher is better
        avg_drift = np.mean(drifts)
        return 1.0 / (1.0 + avg_drift)

    def _compute_coherence(self, logits: List[torch.Tensor]) -> float:
        """Measures entropy of logits over time. Stable entropy = higher coherence."""
        entropies = []
        for l in logits:
            p = F.softmax(l, dim=-1)
            ent = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean().item()
            entropies.append(ent)
        
        # Stability of entropy (lower variance is better)
        ent_std = np.std(entropies) if len(entropies) > 1 else 0.0
        return 1.0 / (1.0 + ent_std)

    def _compute_latent_entropy(self, trajectory: List[torch.Tensor]) -> float:
        """Measures the spatial entropy of the latent states."""
        if not trajectory: return 0.0
        stack = torch.stack(trajectory) # [steps, hidden]
        # Estimate entropy via variance or histogram
        std = stack.std(dim=0).mean().item()
        return std

    def _compute_manifold_smoothness(self, trajectory: List[torch.Tensor]) -> float:
        """Measures acceleration of states (second derivative). Lower is smoother."""
        if len(trajectory) < 3: return 1.0
        accels = []
        for i in range(len(trajectory) - 2):
            v1 = trajectory[i+1] - trajectory[i]
            v2 = trajectory[i+2] - trajectory[i+1]
            accel = torch.norm(v2 - v1, p=2).item()
            accels.append(accel)
        avg_accel = np.mean(accels)
        return 1.0 / (1.0 + avg_accel)

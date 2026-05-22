"""
analysis/collapse_basin_analyzer.py

Analyzes the geometry of collapse basins in the latent manifold.
Estimates basin depth, escape energy, and irreversibility.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

class CollapseBasinAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.history = []
        self.basin_threshold = self.config.get("basin_threshold", 0.7)
        self.energy_scale = self.config.get("energy_scale", 1.0)

    def analyze_trajectory(self, 
                           latent_trajectory: torch.Tensor, 
                           health_scores: torch.Tensor) -> Dict[str, float]:
        """
        Analyzes a sequence of latent states to estimate basin properties.
        latent_trajectory: (steps, hidden_dim)
        health_scores: (steps,)
        """
        steps = latent_trajectory.shape[0]
        if steps < 2:
            return {"basin_depth": 0.0, "escape_probability": 1.0, "irreversibility_score": 0.0}

        # 1. Basin Depth Estimation
        # Depth is proportional to the drop in cognitive health and the persistence of the low-health state
        health_min = health_scores.min().item()
        health_mean = health_scores.mean().item()
        
        # Basin depth increases as health drops and stays low
        basin_depth = 1.0 - health_min
        
        # 2. Escape Energy Estimation
        # Energy required to 'climb out' of the basin. 
        # Calculated as the inverse of the gradient of health recovery.
        health_diffs = torch.diff(health_scores)
        recovery_speed = torch.clamp(health_diffs, min=0).mean().item()
        
        # If recovery speed is low despite interventions, escape energy is high
        escape_energy = 1.0 / (recovery_speed + 1e-6)
        
        # 3. Irreversibility Scoring
        # A region is irreversible if health saturates at low values and latent drift remains high
        recent_health = health_scores[-5:] if steps >= 5 else health_scores
        health_stdev = recent_health.std().item()
        
        # If health is low AND stable (low variance), it's a 'flat' basin bottom (trapped)
        is_trapped = (recent_health.mean() < 0.2) and (health_stdev < 0.05)
        
        # Manifold divergence (L2 distance from start)
        drift = torch.norm(latent_trajectory[-1] - latent_trajectory[0]).item()
        
        irreversibility_score = (basin_depth * 0.5) + (1.0 if is_trapped else 0.0) * 0.5
        irreversibility_score = min(1.0, irreversibility_score * (1.0 + drift * 0.1))

        # 4. Escape Probability
        escape_probability = np.exp(-escape_energy * self.energy_scale)

        return {
            "basin_depth": float(basin_depth),
            "escape_energy": float(escape_energy),
            "escape_probability": float(escape_probability),
            "irreversibility_score": float(irreversibility_score),
            "collapse_persistence": float(steps / (recovery_speed + 0.1)),
            "recovery_difficulty": float(escape_energy * (1.0 + irreversibility_score))
        }

    def detect_attractor_trapping(self, 
                                  latent_trajectory: torch.Tensor, 
                                  window_size: int = 10) -> bool:
        """
        Detects if the system has entered a cyclic or static attractor of collapse.
        """
        if latent_trajectory.shape[0] < window_size:
            return False
        
        recent = latent_trajectory[-window_size:]
        # Calculate pairwise distances in the window
        # If mean distance is small, it's trapped in a point attractor
        # If distances show periodicity, it's a limit cycle
        
        center = recent.mean(dim=0)
        dist_from_center = torch.norm(recent - center, dim=1).mean().item()
        
        # Heuristic: if latent movement is very small but health is low
        return dist_from_center < 0.01

    def map_basin_boundary(self, current_latent: torch.Tensor, perturbed_latents: List[torch.Tensor], model_fn) -> float:
        """
        Estimates the distance to the nearest 'cognitive cliff' or basin boundary.
        """
        # Placeholder for boundary mapping logic
        return 0.5

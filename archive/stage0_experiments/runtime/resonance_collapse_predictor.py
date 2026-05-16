import torch
import torch.nn as nn
from typing import Dict, List, Optional
from anchor_logic.cognitive_guard_network import CognitiveGuardNetwork

class ResonanceCollapsePredictor(CognitiveGuardNetwork):
    """
    Extends CognitiveGuardNetwork to predict resonance fracture.
    Includes synchronization metrics and inter-layer drift propagation.
    """
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64):
        # We increase input_dim to include:
        # 13. Global Coherence Score
        # 14. Synchronization Entropy
        # 15. Max Drift Coupling
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        
        # Add a specific head for resonance fracture
        self.resonance_fracture_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        res = super().forward(x)
        latent = self.encoder(x)
        
        fracture_prob = torch.sigmoid(self.resonance_fracture_head(latent))
        res["resonance_fracture_probability"] = fracture_prob
        return res

    @staticmethod
    def prepare_resonance_input(metrics: Dict[str, float], 
                                resonance_metrics: Dict[str, float],
                                pos: int, 
                                max_pos: int, 
                                repair_count: int, 
                                anchor_count: int) -> torch.Tensor:
        """
        Prepares input vector including resonance metrics.
        """
        # Standard metrics (12 dims)
        vec = [
            metrics.get("latent_velocity", 0.0),
            metrics.get("latent_acceleration", 0.0),
            metrics.get("trajectory_curvature", 0.0),
            metrics.get("hidden_drift", 0.0),
            metrics.get("attention_entropy", 0.0),
            metrics.get("attention_fragmentation", 0.0),
            metrics.get("top_k_overlap", 0.8),
            metrics.get("basin_escape_score", 0.0),
            anchor_count / 100.0,
            repair_count / 10.0,
            pos / max_pos if max_pos > 0 else 0.0,
            metrics.get("sequence_entropy", 1.0),
            
            # Resonance metrics (3 dims)
            resonance_metrics.get("coherence", 1.0),
            resonance_metrics.get("entropy", 0.0),
            resonance_metrics.get("max_drift_coupling", 0.0)
        ]
        return torch.tensor(vec, dtype=torch.float32)

"""
anchor_logic/cognitive_guard_network.py
Phase 16: Learned Cognitive Guards (LCG)
Implements a lightweight guard model to predict reasoning collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

class CognitiveGuardNetwork(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        # Inputs:
        # 1. latent_velocity
        # 2. latent_acceleration
        # 3. trajectory_curvature
        # 4. hidden_state_drift
        # 5. attention_entropy
        # 6. attention_fragmentation
        # 7. top-k overlap (simulated or real)
        # 8. basin_escape_score
        # 9. anchor_density
        # 10. repair_history (count)
        # 11. current_position (normalized)
        # 12. sequence_entropy
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Heads
        self.collapse_head = nn.Linear(hidden_dim, 1) # Probability
        self.urgency_head = nn.Linear(hidden_dim, 1)  # 0 to 1
        self.strategy_head = nn.Linear(hidden_dim, 4) # One-hot: [Noop, Anchor, Boost, Fallback]
        self.survival_head = nn.Linear(hidden_dim, 1) # Predicted steps until collapse
        
    def forward(self, x):
        """
        x: [batch, input_dim]
        """
        latent = self.encoder(x)
        
        collapse_prob = torch.sigmoid(self.collapse_head(latent))
        urgency = torch.sigmoid(self.urgency_head(latent))
        strategy_logits = self.strategy_head(latent)
        predicted_survival = F.softplus(self.survival_head(latent)) # Must be positive
        
        return {
            "collapse_probability": collapse_prob,
            "intervention_urgency": urgency,
            "strategy_logits": strategy_logits,
            "predicted_reasoning_survival": predicted_survival
        }

    @staticmethod
    def prepare_input(metrics: Dict[str, float], 
                      pos: int, 
                      max_pos: int, 
                      repair_count: int, 
                      anchor_count: int) -> torch.Tensor:
        """
        Converts raw metrics into a tensor for the model.
        """
        # Map keys to indices
        vec = [
            metrics.get("latent_velocity", 0.0),
            metrics.get("latent_acceleration", 0.0),
            metrics.get("trajectory_curvature", 0.0),
            metrics.get("hidden_drift", 0.0),
            metrics.get("attention_entropy", 0.0),
            metrics.get("attention_fragmentation", 0.0),
            metrics.get("top_k_overlap", 0.8), # Default to high overlap
            metrics.get("basin_escape_score", 0.0),
            anchor_count / 100.0, # Normalized anchor density
            repair_count / 10.0,  # Normalized repair history
            pos / max_pos if max_pos > 0 else 0.0,
            metrics.get("sequence_entropy", 1.0)
        ]
        return torch.tensor(vec, dtype=torch.float32)

if __name__ == "__main__":
    model = CognitiveGuardNetwork()
    dummy_input = torch.randn(1, 12)
    output = model(dummy_input)
    print("Guard Output:", output)

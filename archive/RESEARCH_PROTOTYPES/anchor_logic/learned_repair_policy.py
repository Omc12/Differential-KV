"""
anchor_logic/learned_repair_policy.py
Phase 16: Learned Repair Policy
Learns optimal sparse intervention strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class LearnedRepairPolicy(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, num_layers: int = 12, num_heads: int = 8):
        super().__init__()
        # Inputs: Guard latent + context features
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy Heads
        self.repair_strength = nn.Linear(hidden_dim, 1) # Intensity of modification
        self.layer_selection = nn.Linear(hidden_dim, num_layers) # Per-layer repair probability
        self.head_selection = nn.Linear(hidden_dim, num_heads)   # Per-head repair probability (simplified as global per-head)
        self.persistence = nn.Linear(hidden_dim, 1) # How many steps to keep boost active
        
    def forward(self, guard_latent: torch.Tensor):
        """
        guard_latent: Latent representation from CognitiveGuardNetwork encoder
        """
        x = self.policy_net(guard_latent)
        
        strength = torch.sigmoid(self.repair_strength(x))
        layer_probs = torch.sigmoid(self.layer_selection(x))
        head_probs = torch.sigmoid(self.head_selection(x))
        persistence_steps = torch.ceil(torch.sigmoid(self.persistence(x)) * 10) # 0 to 10 steps
        
        return {
            "strength": strength,
            "layer_mask": layer_probs,
            "head_mask": head_probs,
            "persistence": persistence_steps
        }

class LearnedRepairController:
    def __init__(self, guard: nn.Module, policy: nn.Module):
        self.guard = guard
        self.policy = policy
        self.current_persistence = 0
        self.active_strategy = None

    def decide_intervention(self, metrics: Dict[str, float], pos: int, max_pos: int, repair_history: int, anchor_count: int):
        input_tensor = self.guard.prepare_input(metrics, pos, max_pos, repair_history, anchor_count).unsqueeze(0)
        
        # Use guard encoder to get latent
        with torch.no_grad():
            latent = self.guard.encoder(input_tensor)
            guard_out = self.guard(input_tensor)
            
            if guard_out["collapse_probability"] > 0.5 or self.current_persistence > 0:
                policy_out = self.policy(latent)
                
                if self.current_persistence == 0:
                    self.current_persistence = policy_out["persistence"].item()
                    self.active_strategy = policy_out
                else:
                    self.current_persistence -= 1
                
                return {
                    "should_intervene": True,
                    "guard": guard_out,
                    "policy": self.active_strategy
                }
            
        return {"should_intervene": False}

if __name__ == "__main__":
    from anchor_logic.cognitive_guard_network import CognitiveGuardNetwork
    guard = CognitiveGuardNetwork()
    policy = LearnedRepairPolicy()
    controller = LearnedRepairController(guard, policy)
    
    metrics = {"latent_velocity": 0.5, "hidden_drift": 0.2}
    decision = controller.decide_intervention(metrics, 10, 100, 0, 5)
    print("Decision:", decision)

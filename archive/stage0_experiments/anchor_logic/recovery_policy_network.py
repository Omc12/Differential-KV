"""
anchor_logic/recovery_policy_network.py

Learned policy model for selecting optimal recovery actions.
Decides between ACTR repair, latent rewinding, or trajectory branching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

class RecoveryPolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 16, action_dim: int = 4):
        super().__init__()
        # State: [health, collapse_prob, drift, acceleration, step_ratio, vram_usage, ...]
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.policy_head = nn.Linear(32, action_dim) # 0: Continue, 1: ACTR_Repair, 2: Rewind, 3: Branch
        self.value_head = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        logits = self.policy_head(x)
        value = self.value_head(x)
        
        return logits, value

class RecoveryPolicyLearner:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = RecoveryPolicyNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.action_map = {0: "CONTINUE", 1: "ACTR_REPAIR", 2: "REWIND", 3: "BRANCH"}
        self.history = []

    def select_action(self, state_dict: Dict[str, float]) -> Tuple[int, str, float]:
        """
        Converts state dict to tensor and selects an action.
        """
        # Feature engineering
        state_vector = torch.tensor([
            state_dict.get("cognitive_health_score", 1.0),
            state_dict.get("collapse_probability", 0.0),
            state_dict.get("latent_drift", 0.0),
            state_dict.get("health_acceleration", 0.0),
            state_dict.get("vram_usage_norm", 0.0),
            state_dict.get("basin_depth", 0.0),
            state_dict.get("escape_energy_norm", 0.0),
            state_dict.get("step_ratio", 0.0),
            # Padding to state_dim=16
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state_vector)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
            
        return action, self.action_map[action], probs[action].item()

    def update_policy(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        """
        Update policy using a simple RL step (Policy Gradient / Actor-Critic).
        """
        self.model.train()
        logits, value = self.model(state)
        
        # Reward components:
        # + Survival (health > 0.5)
        # - VRAM cost
        # - Intervention penalty (we want minimal intervention)
        
        # Placeholder for RL training logic
        pass

    def get_reward(self, health_before: float, health_after: float, action: int, intervention_cost: float) -> float:
        """
        Calculates the reward for an intervention.
        """
        improvement = health_after - health_before
        survival_bonus = 1.0 if health_after > 0.7 else -1.0
        cost_penalty = intervention_cost * 0.5
        
        # Penalize repeated interventions that don't help (death spiral prevention)
        if action != 0 and improvement < 0.05:
            return -2.0
            
        return improvement + survival_bonus - cost_penalty

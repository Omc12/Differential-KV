"""
specialization/cognitive_role_evolution.py

Manages the evolution of cognitive roles within the collective.
"""

import torch
from typing import Dict, List, Optional, Any

class CognitiveRoleEvolution:
    """
    Tracks how agents' roles evolve based on their contributions to the collective.
    Assigns roles like 'Strategist', 'Worker', 'Validator' based on manifold signatures.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_roles = {} # agent_id -> role_metadata
        self.role_definitions = {
            "orchestrator": {"resonance_signature": 0.9, "stability_requirement": 0.95},
            "specialist": {"resonance_signature": 0.7, "stability_requirement": 0.8},
            "explorer": {"resonance_signature": 0.3, "stability_requirement": 0.5},
        }

    def evolve_role(self, agent_id: str, contribution_metrics: Dict[str, float]):
        """
        Updates an agent's role based on their recent performance and signature.
        """
        stability = contribution_metrics.get("stability", 0.5)
        impact = contribution_metrics.get("impact", 0.5)
        
        # Simple role assignment logic
        if stability > 0.9 and impact > 0.8:
            role = "orchestrator"
        elif stability > 0.7:
            role = "specialist"
        else:
            role = "explorer"
            
        self.agent_roles[agent_id] = {
            "role": role,
            "evolution_stage": self.agent_roles.get(agent_id, {}).get("evolution_stage", 0) + 1,
            "metrics": contribution_metrics
        }
        
        return role

    def get_role_distribution(self) -> Dict[str, int]:
        """Returns the count of agents in each role."""
        dist = {}
        for meta in self.agent_roles.values():
            r = meta["role"]
            dist[r] = dist.get(r, 0) + 1
        return dist

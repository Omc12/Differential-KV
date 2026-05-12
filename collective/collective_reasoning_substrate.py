"""
collective/collective_reasoning_substrate.py

The main orchestration layer for collective cognition in Differential KV.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collective.shared_manifold_exchange import SharedManifoldExchange
from collective.collaborative_attractor_pool import CollaborativeAttractorPool

class CollectiveReasoningSubstrate:
    """
    Orchestrates shared reasoning manifolds across multiple cognitive agents.
    Enables collective intelligence through synchronized attractor states.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange = SharedManifoldExchange(config)
        self.attractor_pool = CollaborativeAttractorPool(config)
        self.agents = {} # agent_id -> agent_metadata
        self.shared_stabilization_memory = {} # manifold_id -> stability_metrics
        self.global_resonance_threshold = config.get("global_resonance_threshold", 0.85)

    def register_agent(self, agent_id: str, identity_fingerprint: torch.Tensor):
        """Registers a new cognitive agent into the substrate."""
        self.agents[agent_id] = {
            "fingerprint": identity_fingerprint,
            "active_manifolds": [],
            "contribution_score": 1.0,
            "resonance_history": []
        }

    def synchronize_collective_state(self, manifolds: Dict[str, torch.Tensor]):
        """
        Synchronizes attractors and manifolds across the registered agents.
        """
        # Distribute shared manifolds through the exchange
        sync_results = self.exchange.broadcast_manifolds(manifolds)
        
        # Update attractor pool with new discoveries
        for mid, manifold in manifolds.items():
            self.attractor_pool.update_attractor(mid, manifold)
            
        return sync_results

    def route_cognition(self, task_manifold: torch.Tensor) -> List[str]:
        """
        Routes a reasoning task to the most suitable agents based on resonance.
        """
        candidates = []
        for agent_id, meta in self.agents.items():
            # resonance = torch.cosine_similarity(task_manifold, meta["fingerprint"], dim=-1)
            resonance = 0.9 # Mock resonance for prototype
            if resonance > self.global_resonance_threshold:
                candidates.append(agent_id)
        
        return candidates if candidates else list(self.agents.keys())[:1]

    def update_stabilization_memory(self, manifold_id: str, health_metrics: Dict[str, float]):
        """Updates the shared stabilization memory with agent-specific feedback."""
        if manifold_id not in self.shared_stabilization_memory:
            self.shared_stabilization_memory[manifold_id] = []
        
        self.shared_stabilization_memory[manifold_id].append(health_metrics)
        # Prune old metrics
        if len(self.shared_stabilization_memory[manifold_id]) > 100:
            self.shared_stabilization_memory[manifold_id].pop(0)

    def get_collective_health(self) -> float:
        """Returns the overall health score of the collective reasoning substrate."""
        if not self.shared_stabilization_memory:
            return 1.0
        
        scores = []
        for metrics_list in self.shared_stabilization_memory.values():
            for m in metrics_list:
                scores.append(m.get("health_score", 1.0))
        
        return sum(scores) / len(scores) if scores else 1.0

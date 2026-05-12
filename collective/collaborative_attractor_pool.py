"""
collective/collaborative_attractor_pool.py

Manages a shared pool of collaborative attractors that guide reasoning.
"""

import torch
from typing import Dict, List, Optional, Any

class CollaborativeAttractorPool:
    """
    Maintains a pool of stable reasoning attractors discovered by the collective.
    Allows agents to pull relevant attractors to stabilize their local manifolds.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.attractors = {} # attractor_id -> metadata
        self.resonance_threshold = config.get("pool_resonance_threshold", 0.9)

    def update_attractor(self, attractor_id: str, tensor: torch.Tensor):
        """Updates or adds an attractor to the collaborative pool."""
        if attractor_id not in self.attractors:
            self.attractors[attractor_id] = {
                "tensor": tensor,
                "stability_count": 1,
                "agents_using": 0,
                "last_sync": 0
            }
        else:
            # Momentum-based update of the attractor geometry
            alpha = 0.1
            self.attractors[attractor_id]["tensor"] = (
                (1 - alpha) * self.attractors[attractor_id]["tensor"] + alpha * tensor
            )
            self.attractors[attractor_id]["stability_count"] += 1

    def find_resonance(self, local_manifold: torch.Tensor) -> List[str]:
        """Finds attractors in the pool that resonate with a local manifold."""
        matches = []
        for aid, meta in self.attractors.items():
            # In real implementation, perform vectorized similarity
            # sim = torch.cosine_similarity(local_manifold, meta["tensor"], dim=-1).max()
            sim = 0.95 # Mock
            if sim > self.resonance_threshold:
                matches.append(aid)
        return matches

    def synchronize_attractors(self, active_ids: List[str]):
        """Marks attractors as active to prevent pruning."""
        for aid in active_ids:
            if aid in self.attractors:
                self.attractors[aid]["agents_using"] += 1

    def prune_unstable_attractors(self):
        """Removes attractors that fail to stabilize the collective reasoning."""
        to_remove = []
        for aid, meta in self.attractors.items():
            if meta["stability_count"] < 2 and meta["agents_using"] == 0:
                to_remove.append(aid)
        
        for aid in to_remove:
            del self.attractors[aid]

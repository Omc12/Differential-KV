"""
collective/shared_manifold_exchange.py

Handles the exchange of reasoning manifolds between distributed cognitive agents.
"""

import torch
from typing import Dict, List, Optional, Any

class SharedManifoldExchange:
    """
    Protocol for efficient manifold inheritance and transfer.
    Supports delta-encoded manifold updates to save bandwidth.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manifold_registry = {} # manifold_id -> (tensor, version)
        self.transfer_stats = {"bytes_exchanged": 0, "manifolds_transferred": 0}

    def broadcast_manifolds(self, manifolds: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """Broadcasts multiple manifolds to the collective registry."""
        results = {}
        for mid, manifold in manifolds.items():
            results[mid] = self.update_manifold(mid, manifold)
        return results

    def update_manifold(self, manifold_id: str, manifold_data: torch.Tensor) -> bool:
        """Updates a specific manifold in the registry with version tracking."""
        if manifold_id in self.manifold_registry:
            old_data, version = self.manifold_registry[manifold_id]
            # Check for significant change before incrementing version
            drift = torch.norm(manifold_data - old_data) if manifold_data.shape == old_data.shape else 1.0
            if drift > 0.05:
                self.manifold_registry[manifold_id] = (manifold_data, version + 1)
        else:
            self.manifold_registry[manifold_id] = (manifold_data, 1)
        
        self.transfer_stats["manifolds_transferred"] += 1
        return True

    def request_manifold(self, manifold_id: str) -> Optional[torch.Tensor]:
        """Retrieves a manifold from the shared registry."""
        if manifold_id in self.manifold_registry:
            return self.manifold_registry[manifold_id][0]
        return None

    def get_exchange_metrics(self) -> Dict[str, Any]:
        """Returns metrics about manifold exchange performance."""
        return {
            "registry_size": len(self.manifold_registry),
            "total_transfers": self.transfer_stats["manifolds_transferred"],
            "average_version": sum(v for _, v in self.manifold_registry.values()) / max(1, len(self.manifold_registry))
        }

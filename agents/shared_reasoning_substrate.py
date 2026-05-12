import torch
from typing import Dict, List, Optional, Any
import os

class SharedReasoningSubstrate:
    """
    A distributed substrate for sharing reasoning manifolds and motifs across agents.
    Enables persistent cognition inheritance and collaborative attractor evolution.
    """
    def __init__(self, substrate_dir: str = "shared_substrate"):
        self.substrate_dir = substrate_dir
        os.makedirs(substrate_dir, exist_ok=True)
        self.shared_pools = {} # category -> list of motifs

    def publish_motifs(self, agent_id: str, category: str, motifs: torch.Tensor):
        """
        Publishes an agent's stable motifs to the shared substrate.
        """
        if category not in self.shared_pools:
            self.shared_pools[category] = []
            
        # Store with agent metadata
        entry = {
            "agent_id": agent_id,
            "motifs": motifs.detach().cpu(),
            "timestamp": "now"
        }
        self.shared_pools[category].append(entry)
        
        # Save to disk for persistence
        path = os.path.join(self.substrate_dir, f"{category}_pool.pt")
        torch.save(self.shared_pools[category], path)

    def retrieve_motifs(self, category: str) -> List[torch.Tensor]:
        """
        Retrieves all shared motifs for a specific category.
        """
        if category not in self.shared_pools:
            path = os.path.join(self.substrate_dir, f"{category}_pool.pt")
            if os.path.exists(path):
                self.shared_pools[category] = torch.load(path)
            else:
                return []
                
        return [entry["motifs"] for entry in self.shared_pools[category]]

    def get_substrate_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the shared substrate.
        """
        return {
            "categories": list(self.shared_pools.keys()),
            "total_motifs": sum(len(pool) for pool in self.shared_pools.values())
        }

import torch
from typing import Dict, List

class AdaptiveSyncPolicies:
    """
    Manages synchronization policies for distributed cognitive runtime.
    """
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.node_entropy = torch.zeros(n_nodes)
        self.sync_frequency = torch.ones(n_nodes) # Frequency scale
        
    def update_node_state(self, node_id: int, entropy: float):
        self.node_entropy[node_id] = entropy
        
    def get_sync_schedule(self) -> Dict[int, bool]:
        """
        Determines which nodes need to synchronize in the current step.
        Nodes with high entropy or high drift get priority.
        """
        mean_entropy = self.node_entropy.mean()
        schedule = {}
        
        for i in range(self.n_nodes):
            # If node entropy is significantly higher than mean, force sync
            if self.node_entropy[i] > mean_entropy * 1.2:
                schedule[i] = True
                self.sync_frequency[i] = min(10.0, self.sync_frequency[i] * 1.1)
            else:
                # Probabilistic sync based on entropy
                prob = self.node_entropy[i] / (mean_entropy + 1e-6)
                schedule[i] = torch.rand(1).item() < (prob * 0.1)
                self.sync_frequency[i] = max(1.0, self.sync_frequency[i] * 0.95)
                
        return schedule

    def get_scaling_factors(self) -> torch.Tensor:
        """Returns scaling factors for resonance per node."""
        return 1.0 + self.node_entropy * 0.5

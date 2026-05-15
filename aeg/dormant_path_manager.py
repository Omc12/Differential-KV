
import torch
import time
from typing import Dict, List, Set, Any

class DormantPathManager:
    """
    PHASE 22.1: AEG - Dormant Path Manager.
    Tracks execution branches that have not been activated for a duration.
    """
    def __init__(self, hibernation_threshold: int = 16):
        self.hibernation_threshold = hibernation_threshold
        self.inactivity_counters = {} # node_idx -> count
        self.hibernated_nodes: Set[int] = set()
        
    def update_activity(self, active_mask: torch.Tensor):
        """
        Updates inactivity counters and determines which paths to hibernate.
        """
        for i, active in enumerate(active_mask):
            if active > 0.5:
                self.inactivity_counters[i] = 0
                if i in self.hibernated_nodes:
                    self.hibernated_nodes.remove(i)
            else:
                self.inactivity_counters[i] = self.inactivity_counters.get(i, 0) + 1
                
                if self.inactivity_counters[i] >= self.hibernation_threshold:
                    self.hibernated_nodes.add(i)

    def filter_activations(self, proposed_activations: torch.Tensor) -> torch.Tensor:
        """
        Suppresses activations for hibernated nodes unless they show strong recovery.
        """
        filtered = proposed_activations.clone()
        for i in self.hibernated_nodes:
            # If proposed activation is weak, kill it (hibernation)
            if filtered[i] < 0.6: # Threshold to wake up
                filtered[i] = 0.0
        return filtered

    def get_dormant_ratio(self) -> float:
        if not self.inactivity_counters:
            return 0.0
        return len(self.hibernated_nodes) / len(self.inactivity_counters)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "dormant_path_ratio": self.get_dormant_ratio(),
            "hibernated_node_count": len(self.hibernated_nodes)
        }

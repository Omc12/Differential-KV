
import torch
from typing import Dict, List, Any

class ExecutionDependencyMapper:
    """
    PHASE 22.1: AEG - Execution Dependency Mapper.
    Analyzes layer/compute dependencies during runtime propagation.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.dependency_counts = torch.zeros(num_layers, num_layers)
        self.total_steps = 0

    def map_step_dependencies(self, 
                              active_layers: torch.Tensor, 
                              symbolic_impact: torch.Tensor):
        """
        Records which layers were active together or triggered by symbolic events.
        active_layers: binary mask [num_layers]
        symbolic_impact: scores [num_layers]
        """
        # Outer product to find co-activation patterns
        co_activation = torch.outer(active_layers.float(), active_layers.float())
        
        # Weight by symbolic impact (if layer A and B are active and B has high impact, 
        # A likely depends on B for this context)
        impact_weighted = co_activation * symbolic_impact.unsqueeze(0)
        
        # Accumulate
        momentum = 0.95
        self.dependency_counts = momentum * self.dependency_counts + (1 - momentum) * impact_weighted
        self.total_steps += 1

    def get_refined_topology(self) -> torch.Tensor:
        """
        Returns a normalized dependency matrix for the execution graph.
        """
        if self.total_steps == 0:
            return torch.eye(self.num_layers)
            
        # Normalize by row/column or just globally
        norms = self.dependency_counts.max()
        if norms > 0:
            return self.dependency_counts / norms
        return torch.eye(self.num_layers)

    def get_bottleneck_layers(self) -> List[int]:
        """
        Identifies layers that are high-dependency hubs.
        """
        hub_scores = self.dependency_counts.sum(dim=0)
        threshold = hub_scores.mean() + hub_scores.std()
        return torch.where(hub_scores > threshold)[0].tolist()


import torch
from typing import Dict, Any

class GraphStabilityController:
    """
    PHASE 22.1: AEG - Graph Stability Controller.
    Governs activation cascades and prevents sparse fragmentation.
    """
    def __init__(self, max_activation_drift: float = 0.2):
        self.max_activation_drift = max_activation_drift
        self.last_global_activation = 0.5
        self.cascade_count = 0
        self.suppression_events = 0

    def suppress_cascade(self, proposed_activation: torch.Tensor) -> torch.Tensor:
        """
        Intervenes if proposed activation levels are jumping too wildly (cascade).
        """
        current_mean = proposed_activation.mean().item()
        drift = abs(current_mean - self.last_global_activation)
        
        if drift > self.max_activation_drift:
            self.cascade_count += 1
            # Apply dampening
            dampening = self.max_activation_drift / (drift + 1e-9)
            stable_activation = self.last_global_activation + (current_mean - self.last_global_activation) * dampening
            
            # Scale the entire tensor to match stable mean
            scale = stable_activation / (current_mean + 1e-9)
            proposed_activation = proposed_activation * scale
            self.suppression_events += 1
            
        self.last_global_activation = proposed_activation.mean().item()
        return torch.clamp(proposed_activation, 0, 1)

    def check_fragmentation(self, activation_mask: torch.Tensor) -> bool:
        """
        Checks if activation is too fragmented (checkered/unstable).
        """
        # Simple measure: count transitions in the mask
        transitions = (activation_mask[:-1] != activation_mask[1:]).sum().item()
        # If transitions exceed 50% of nodes, it's fragmented
        return transitions > (len(activation_mask) // 2)

    def get_stability_health(self) -> float:
        """
        Returns 1.0 (stable) to 0.0 (unstable).
        """
        penalty = (self.cascade_count * 0.1 + self.suppression_events * 0.05)
        return max(0.0, 1.0 - (penalty / 10.0)) # Normalized over a window

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "cascade_suppression_health": self.get_stability_health(),
            "suppression_events": self.suppression_events,
            "mean_activation_drift": abs(self.last_global_activation - 0.5) # relative to center
        }

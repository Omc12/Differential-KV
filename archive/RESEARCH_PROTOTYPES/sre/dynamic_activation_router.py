
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class DynamicActivationRouter:
    """
    PHASE 22.0: SRE - Dynamic Activation Router.
    Selective execution activation based on symbolic compute localization.
    """
    def __init__(self, num_layers: int, activation_threshold: float = 0.3):
        self.num_layers = num_layers
        self.activation_threshold = activation_threshold
        self.activation_history: List[torch.Tensor] = []
        
        # Adaptive focus parameters
        self.layer_focus = torch.ones(num_layers) / num_layers
        self.momentum = 0.9

    def route_activation(self, symbolic_context_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculates activation weights for transformer participation paths.
        symbolic_context_vector: High-level representation of current symbolic state.
        Returns: Activation weights per layer [num_layers].
        """
        # Projects symbolic context to layer-wise activation scores
        # In a real impl, this might be a small MLP or linear projection
        # For validation, we use a similarity-based approach
        
        # Mock projection: symbolic importance tends to peak in middle-to-late layers
        # for complex reasoning, and early layers for structural recognition.
        scores = torch.zeros(self.num_layers)
        
        # Heuristic: if symbolic context is "dense", activate more layers
        entropy = -torch.sum(symbolic_context_vector * torch.log(symbolic_context_vector + 1e-9))
        
        for i in range(self.num_layers):
            # Dynamic focus adjustment
            dist_to_peak = abs(i - self.num_layers // 2)
            base_activation = torch.exp(-torch.tensor(dist_to_peak / 5.0))
            
            # Modulate by symbolic entropy (high entropy -> need more layers)
            scores[i] = base_activation * (1.0 + entropy.item() * 0.2)
            
        # Normalize and apply threshold
        scores = torch.sigmoid(scores - 0.5)
        
        # Apply adaptive momentum
        self.layer_focus = self.momentum * self.layer_focus + (1 - self.momentum) * scores
        
        return self.layer_focus

    def get_participation_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Determines which layers should participate in the current execution step.
        """
        return scores > self.activation_threshold

    def adapt_runtime_focus(self, symbolic_error: float):
        """
        Adjusts activation threshold if symbolic continuity is breaking.
        """
        if symbolic_error > 0.2:
            # Drop threshold to activate more regions if we are failing
            self.activation_threshold = max(0.1, self.activation_threshold - 0.05)
        else:
            # Increase threshold to save compute if we are stable
            self.activation_threshold = min(0.6, self.activation_threshold + 0.01)
            
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "mean_activation": self.layer_focus.mean().item(),
            "active_layers": (self.layer_focus > self.activation_threshold).sum().item(),
            "threshold": self.activation_threshold
        }


import torch
from typing import Dict, Any

class EntropyStabilityRegulator:
    """
    PHASE 22.4: ARO - Entropy Stability Regulator.
    Protects entropy diversity to prevent deterministic optimization collapse.
    """
    def __init__(self, target_entropy: float = 0.5):
        self.target_entropy = target_entropy
        self.diversity_health = 1.0

    def regulate_optimization(self, 
                               proposed_activation: torch.Tensor) -> torch.Tensor:
        """
        Intervenes if proposed activations are becoming too deterministic (low entropy).
        """
        # Calculate current entropy of the activation mask
        probs = torch.clamp(proposed_activation, 0.001, 0.999)
        entropy = -torch.mean(probs * torch.log(probs) + (1-probs) * torch.log(1-probs))
        
        self.diversity_health = entropy.item() / self.target_entropy
        
        if self.diversity_health < 0.6: # Deterministic collapse warning
            # Inject structured noise to restore diversity
            noise = torch.rand_like(proposed_activation) * 0.1
            return torch.clamp(proposed_activation + noise, 0, 1)
            
        return proposed_activation

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "entropy_diversity_health": self.diversity_health,
            "is_collapsing": self.diversity_health < 0.5
        }

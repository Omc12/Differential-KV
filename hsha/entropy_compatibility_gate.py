
import torch
import torch.nn.functional as F
from typing import Optional

class EntropyCompatibilityGate:
    """
    PHASE 21.1: Preserves decoder diversity during symbolic recall.
    Prevents deterministic collapse by monitoring and gating the injection strength.
    """
    def __init__(self, min_entropy: float = 0.3):
        self.min_entropy = min_entropy
        self.last_entropy = 1.0

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculates Shannon entropy of the logit distribution."""
        probs = F.softmax(logits, dim=-1)
        # Avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()
        self.last_entropy = entropy
        return entropy

    def blend_softly(self, original_logits: torch.Tensor, 
                     target_id: int, boost: float) -> torch.Tensor:
        """
        Softly blends a symbolic boost into the original distribution.
        If the resulting entropy would be too low, the boost is scaled down.
        """
        # Test the boost
        test_logits = original_logits.clone()
        if target_id < test_logits.shape[-1]:
            test_logits[0, target_id] += boost
            
        test_entropy = self.calculate_entropy(test_logits)
        
        # If entropy collapses, dampen the boost
        # We allow some entropy reduction for symbolic stability, but not a collapse to 0.
        if test_entropy < self.min_entropy:
            # Dampening factor
            factor = (test_entropy + 1e-4) / (self.min_entropy + 1e-4)
            boost *= factor
            
            # Apply final boost
            final_logits = original_logits.clone()
            final_logits[0, target_id] += boost
            return final_logits
            
        return test_logits

    def is_safe_to_inject(self) -> bool:
        """Simple safety check based on last known entropy."""
        return self.last_entropy > self.min_entropy

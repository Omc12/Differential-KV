
import torch
import torch.nn.functional as F
from typing import Set

class ProbabilisticTopologyBlender:
    """
    PHASE 21.3: STRL - Probabilistic Topology Blender.
    Softly blends structural repair signals into the distribution.
    """
    def __init__(self, repair_strength: float = 0.5):
        self.repair_strength = repair_strength

    def blend_repair(self, logits: torch.Tensor, target_ids: Set[int], drift: float) -> torch.Tensor:
        """
        Gently boosts structural tokens (delimiters) when drift is detected.
        Maintains entropy by using soft probabilistic blending.
        """
        if drift < 0.1 or not target_ids:
            return logits
            
        blended_logits = logits.clone()
        effective_boost = self.repair_strength * drift
        
        target_indices = torch.tensor(list(target_ids), device=logits.device)
        target_indices = target_indices[target_indices < logits.shape[-1]]
        
        if len(target_indices) > 0:
            blended_logits[0, target_indices] += effective_boost
            
        return blended_logits

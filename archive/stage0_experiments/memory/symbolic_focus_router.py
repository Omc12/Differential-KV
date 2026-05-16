
import torch
from typing import List, Optional

class SymbolicFocusRouter:
    """
    Phase 20.8: Redirects attention toward active symbolic lineages and anchors.
    Triggered when drift risk increases or attention fragmentation is high.
    """
    def __init__(self, base_redirection_strength: float = 5.0):
        self.strength = base_redirection_strength

    def route_focus(self, logits: torch.Tensor, target_token_ids: List[int], drift_risk: float) -> torch.Tensor:
        """
        Redirects logit mass toward target tokens based on risk.
        - logits: [1, vocab_size]
        - target_token_ids: tokens to boost
        - drift_risk: 0.0 to 1.0 (measured by integrity field or fragmentation)
        """
        if not target_token_ids:
            return logits
            
        # Redirection is proportional to risk
        redirection_bias = self.strength * drift_risk
        
        # Apply bias
        t_idx = torch.tensor(target_token_ids, device=logits.device)
        t_idx = t_idx[t_idx < logits.shape[-1]]
        
        if len(t_idx) > 0:
            logits[0, t_idx] += redirection_bias
            
        return logits

    def calculate_drift_risk(self, fragmentation: float, integrity_drift: bool) -> float:
        """Heuristic risk calculation."""
        risk = 0.0
        if integrity_drift:
            risk += 0.6
        
        # Fragmentation risk (normalized roughly)
        frag_risk = min(0.4, (fragmentation - 4.0) / 10.0) if fragmentation > 4.0 else 0.0
        risk += frag_risk
        
        return min(1.0, risk)

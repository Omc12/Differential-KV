import torch
from typing import List, Dict, Optional

class SymbolicPrecisionField:
    """
    SPS Phase 20.6: Symbolic Precision Stabilization Field.
    Creates localized probabilistic reinforcement zones for symbolic tokens.
    
    Reinforces:
    - Digits (0-9)
    - Delimiters (-, _, :, .)
    - Casing (Upper/Lower match)
    - Alphanumeric continuity
    """
    def __init__(self, tokenizer, base_precision_strength: float = 2.0):
        self.tokenizer = tokenizer
        self.base_strength = base_precision_strength
        
        # Token category caches
        self.digit_tokens = self._get_tokens_by_predicate(lambda s: s.strip().isdigit())
        self.delimiter_tokens = self._get_tokens_by_predicate(lambda s: any(d in s for d in "-_:.@/\\"))
        
    def _get_tokens_by_predicate(self, predicate) -> torch.Tensor:
        matching = []
        for i in range(len(self.tokenizer)):
            s = self.tokenizer.decode([i])
            if predicate(s):
                matching.append(i)
        return torch.tensor(matching, device="cuda")

    def get_precision_logits(
        self, 
        expected_token_id: int, 
        current_logits: torch.Tensor,
        stabilization_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Calculates a precision reinforcement vector.
        Does NOT replace logits; provides a bias field.
        """
        precision_bias = torch.zeros_like(current_logits)
        
        if expected_token_id < 0:
            return precision_bias
            
        # 1. Direct match reinforcement (Soft)
        precision_bias[0, expected_token_id] += self.base_strength * stabilization_factor
        
        # 2. Category reinforcement
        # If we expect a digit, slightly boost all digits to prevent alpha-drift
        expected_str = self.tokenizer.decode([expected_token_id]).strip()
        
        if expected_str.isdigit():
            precision_bias[0, self.digit_tokens] += (self.base_strength * 0.25) * stabilization_factor
            
        if any(d in expected_str for d in "-_:.@/\\"):
            precision_bias[0, self.delimiter_tokens] += (self.base_strength * 0.25) * stabilization_factor
            
        return precision_bias

    def estimate_drift_risk(self, token_id: int, expected_id: int) -> float:
        """
        Calculates how 'far' a token is from the expected symbolic identity.
        """
        if token_id == expected_id:
            return 0.0
            
        s1 = self.tokenizer.decode([token_id]).strip()
        s2 = self.tokenizer.decode([expected_id]).strip()
        
        # Casing drift
        if s1.lower() == s2.lower() and s1 != s2:
            return 0.5
            
        # Category drift (Digit vs Non-digit)
        if s1.isdigit() != s2.isdigit():
            return 1.0
            
        return 0.8

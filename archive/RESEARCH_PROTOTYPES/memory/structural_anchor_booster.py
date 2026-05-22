
import torch
from typing import Set, Dict, Optional, List

class StructuralAnchorBooster:
    """
    Phase 20.8: Dynamically reinforces delimiters and symbolic roots.
    Prevents Attention Mass Dilution by amplifying focus on structural pivots.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Core structural delimiters for various symbolic formats
        self.delimiter_chars = {"-", "_", ":", "/", ".", "=", "{", "}", "[", "]", "(", ")", ",", ";", "@", "#", "$", "%", "^", "*", "+", "|", "<", ">"}
        self.delimiter_ids = self._initialize_delimiters()
        
        # Boost configuration
        self.base_boost = 12.0
        self.delimiter_boost = 18.0 # Stronger boost for structural stability
        self.root_boost = 20.0      # Maximum boost for the origin of the lineage

    def _initialize_delimiters(self) -> Set[int]:
        d_ids = set()
        for char in self.delimiter_chars:
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            d_ids.update(ids)
        return d_ids

    def get_boost_vector(self, vocab_size: int, device: torch.device, active_root_id: Optional[int] = None) -> torch.Tensor:
        """Generates a logit bias vector for structural reinforcement."""
        bias = torch.zeros(vocab_size, device=device)
        
        # 1. Boost all structural delimiters
        d_indices = torch.tensor(list(self.delimiter_ids), device=device)
        d_indices = d_indices[d_indices < vocab_size]
        bias[d_indices] = self.delimiter_boost
        
        # 2. Boost the active symbolic root if provided
        if active_root_id is not None and active_root_id < vocab_size:
            bias[active_root_id] = self.root_boost
            
        return bias

class DelimiterIntegrityField:
    """
    Phase 20.8: Tracks structural delimiter health in the output stream.
    Triggers 'Anchor Boosting' when delimiter drift is detected.
    """
    def __init__(self, booster: StructuralAnchorBooster):
        self.booster = booster
        self.consecutive_matches = 0
        self.drift_detected = False

    def update(self, token_id: int, is_expected_delimiter: bool):
        if is_expected_delimiter:
            self.consecutive_matches += 1
            self.drift_detected = False
        else:
            if self.consecutive_matches > 0:
                # Drift detected on an expected delimiter!
                self.drift_detected = True
            self.consecutive_matches = 0

    def get_amplification_factor(self) -> float:
        """Returns a multiplier for boosting if integrity is compromised."""
        return 2.5 if self.drift_detected else 1.0

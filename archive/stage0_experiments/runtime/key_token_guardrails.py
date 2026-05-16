import torch
from typing import List, Pattern
import re

class KeyTokenGuardrails:
    """
    Prevents pruning of critical semantic tokens.
    Uses heuristics and pattern matching to identify 'must-keep' tokens.
    """
    def __init__(self, protected_patterns: List[str] = None):
        self.protected_patterns = protected_patterns or [
            r"system", r"instruction", r"Task:", r"ID:[0-9]+"
        ]
        self.compiled_patterns = [re.compile(p) for p in self.protected_patterns]

    def get_protected_indices(self, tokens: List[str]) -> torch.Tensor:
        """
        Scan tokens for protected patterns.
        """
        indices = []
        for i, token in enumerate(tokens):
            for pattern in self.compiled_patterns:
                if pattern.search(token):
                    indices.append(i)
                    break
        return torch.tensor(indices, dtype=torch.long)

    def validate_pruning(self, pruned_indices: torch.Tensor, protected_indices: torch.Tensor) -> torch.Tensor:
        """
        Ensure no protected tokens are in the pruned set.
        """
        # Remove protected indices from the pruned set
        mask = torch.ones_like(pruned_indices, dtype=torch.bool)
        for idx in protected_indices:
            mask = mask & (pruned_indices != idx)
            
        return pruned_indices[mask]

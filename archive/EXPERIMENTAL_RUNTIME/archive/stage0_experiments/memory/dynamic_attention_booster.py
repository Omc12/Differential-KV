import torch
from typing import List, Set, Optional

class DynamicAttentionBooster:
    """
    PHASE 20.3: Active Attention Steering.
    Injects salience 'boosts' into the decoding process to force 
    the model to attend to preserved symbolic spans.
    """
    def __init__(self, boost_factor: float = 2.5):
        self.boost_factor = boost_factor
        self.active_spans = [] # List of (start_idx, end_idx) in absolute positions
        self.locked_tokens = set()
        self.ordered_sequences = [] # List of lists of token IDs

    def add_span(self, start_idx: int, end_idx: int, tokens: torch.Tensor = None):
        self.active_spans.append((start_idx, end_idx))
        if tokens is not None:
            ids = tokens.flatten().tolist()
            self.ordered_sequences.append(ids)
            for t in ids:
                self.locked_tokens.add(t)

    def get_steering_bias(self, absolute_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Calculates a bias vector for the given absolute indices.
        """
        bias = torch.ones_like(absolute_indices, dtype=torch.float, device=device)
        for start, end in self.active_spans:
            mask = (absolute_indices >= start) & (absolute_indices <= end)
            bias[mask] *= self.boost_factor
        return bias

    def get_locked_token_ids(self):
        # For backward compatibility, return all unique locked tokens
        return list(self.locked_tokens)

    def get_ordered_symbolic_sequence(self, recent_tokens: List[int] = None):
        """Returns the sequence that best matches the recent generated tokens."""
        if not self.ordered_sequences:
            return []
            
        if not recent_tokens:
            return self.ordered_sequences[0]
            
        # Find match in any sequence
        for seq in self.ordered_sequences:
            if len(seq) < len(recent_tokens):
                continue
            # Simple window check
            for i in range(len(seq) - len(recent_tokens) + 1):
                if seq[i:i+len(recent_tokens)] == recent_tokens:
                    return seq
                    
        return self.ordered_sequences[0]

    def reset(self):
        self.active_spans = []
        self.locked_tokens = set()
        self.ordered_sequences = []

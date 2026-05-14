import torch

class LocalTransitionExpander:
    """
    PHASE 19.0B: Local Transition Expander.
    Expands sparse transition boundaries based on local semantic entropy 
    to reduce pruning cliffs.
    """
    def __init__(self, base_expansion: int = 16, entropy_scale: float = 2.0):
        self.base_expansion = base_expansion
        self.entropy_scale = entropy_scale

    def expand_boundaries(self, mask: torch.Tensor, local_entropy: torch.Tensor) -> torch.Tensor:
        """
        Dynamically expands mask boundaries based on local entropy.
        High entropy regions get wider runways.
        """
        if not mask.any():
            return mask
            
        seq_len = mask.shape[1]
        expanded_mask = mask.clone()
        
        # Calculate dynamic expansion width per token
        # local_entropy assumed to be [1, seq_len]
        expansion_widths = (self.base_expansion + self.entropy_scale * local_entropy).long()
        
        # Simple implementation: for each active token, set its neighborhood in expanded_mask
        active_indices = torch.where(mask[0])[0]
        for idx in active_indices:
            width = expansion_widths[0, idx].item()
            start = max(0, idx - width)
            end = min(seq_len, idx + width)
            expanded_mask[0, start:end] = True
            
        return expanded_mask

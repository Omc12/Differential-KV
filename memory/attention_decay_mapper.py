import torch

class AttentionDecayMapper:
    """
    PHASE 19.0C: Attention Decay Mapper.
    Maps temporal distance to attention decay profiles that preserve 
    continuity.
    """
    def __init__(self, decay_type: str = "power_law"):
        self.decay_type = decay_type

    def get_decay_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a decay profile for the given sequence length.
        """
        positions = torch.arange(seq_len, device=device).float()
        # Distance from current (end of sequence)
        distances = seq_len - 1 - positions
        
        if self.decay_type == "power_law":
            # Typical for transformers (Log-Linear)
            weights = 1.0 / (distances + 1.0)**0.5
        elif self.decay_type == "exponential":
            weights = torch.exp(-0.001 * distances)
        else:
            weights = torch.ones_like(positions)
            
        return weights.unsqueeze(0) # [1, seq_len]

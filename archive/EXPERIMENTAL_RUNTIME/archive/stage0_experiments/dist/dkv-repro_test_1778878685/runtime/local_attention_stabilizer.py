import torch
import torch.nn as nn

class LocalAttentionStabilizer(nn.Module):
    """
    Stabilizes attention across adjacent layers to reduce noise and improve retrieval.
    Replaces global 'resonance' with local, bounded smoothing.
    """
    def __init__(self, smoothing_factor: float = 0.1):
        super().__init__()
        self.smoothing_factor = smoothing_factor

    def stabilize(self, current_attn: torch.Tensor, prev_layer_attn: torch.Tensor) -> torch.Tensor:
        """
        Smooths attention weights using the previous layer's distribution.
        current_attn: [batch, heads, q_len, k_len]
        """
        if prev_layer_attn is None:
            return current_attn
            
        # Exponential moving average style stabilization
        stabilized_attn = (1.0 - self.smoothing_factor) * current_attn + self.smoothing_factor * prev_layer_attn
        return stabilized_attn

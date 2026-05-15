
import torch
import numpy as np
from typing import Dict, List, Optional

class AttentionEnergyCompressor:
    """
    Phase 20.8: Reduces symbolic attention diffusion.
    Compresses symbolic attention neighborhoods and preserves local symbolic energy density.
    """
    def __init__(self, compression_factor: float = 1.2):
        self.compression_factor = compression_factor
        self.active = True

    def compress_neighborhoods(self, attention_mass: torch.Tensor, symbolic_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies a non-linear scaling to 'compress' attention toward symbolic peaks.
        - attention_mass: [kv_len]
        - symbolic_mask: [kv_len]
        """
        if not self.active:
            return attention_mass
            
        # Identify symbolic regions
        symbolic_mass = attention_mass * (symbolic_mask > 0).float()
        
        # Apply sharpening (compression) to symbolic mass
        # High-mass areas get amplified more than low-mass areas within the symbolic neighborhood
        compressed_mass = torch.pow(symbolic_mass, self.compression_factor)
        
        # Normalize to preserve total symbolic energy
        orig_total = symbolic_mass.sum()
        new_total = compressed_mass.sum()
        
        if new_total > 1e-9:
            compressed_mass = compressed_mass * (orig_total / new_total)
            
        # Re-inject into non-symbolic mass (which remains untouched or slightly suppressed)
        final_mass = attention_mass.clone()
        final_mass[symbolic_mask > 0] = compressed_mass
        
        return final_mass

    def get_focus_multiplier(self, fragmentation: float) -> float:
        """Determines how much to amplify based on diffusion."""
        # If fragmentation is high, we need more compression
        if fragmentation > 7.0:
            return 1.5
        if fragmentation > 5.0:
            return 1.2
        return 1.0

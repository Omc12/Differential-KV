import torch
import torch.nn as nn
from typing import Dict, List, Optional
from anchor_logic.resonance_anchors import ResonanceAnchor

class ResonanceController:
    """
    Stabilizes individual layers by applying resonance restoration.
    If a layer diverges, it pulls it back towards the synchronized manifold of neighbors.
    """
    def __init__(self, layer_idx: int, dim: int):
        self.layer_idx = layer_idx
        self.dim = dim
        self.restoration_strength = 0.1
        
    def apply_stabilization(self, 
                            current_hidden: torch.Tensor, 
                            neighbor_anchors: List[ResonanceAnchor]) -> torch.Tensor:
        """
        Applies resonance-based restoration to the hidden state.
        Uses phase-aligned projections from neighboring anchors.
        """
        if not neighbor_anchors:
            return current_hidden
            
        # Target state is the weighted average of phase-aligned neighbor states
        # (Simplified: using the latent_directions/phase_state if available)
        restoration_target = torch.zeros_like(current_hidden)
        total_weight = 0.0
        
        for anchor in neighbor_anchors:
            if anchor.phase_state is not None:
                # Project current state onto anchor's resonance phase
                weight = anchor.sync_coherence
                # Simple restoration: mix towards the anchor's phase state
                # Note: In a real model, this would involve a more complex rotation
                restoration_target += anchor.phase_state.to(current_hidden.device) * weight
                total_weight += weight
                
        if total_weight > 0:
            restoration_target /= total_weight
            # Apply restoration
            stabilized = (1.0 - self.restoration_strength) * current_hidden + \
                         self.restoration_strength * restoration_target
            return stabilized
            
        return current_hidden

    def set_strength(self, strength: float):
        self.restoration_strength = max(0.0, min(1.0, strength))

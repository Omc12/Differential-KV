"""
anchor_logic/basin_reinforcement.py
Phase 18: Evolutionary Manifold Shaping
Reinforces stable latent regions by increasing precision and suppressing instability.
"""

import torch
from typing import List, Dict, Any, Tuple

class BasinReinforcementSystem:
    def __init__(self, stabilization_factor: float = 1.2):
        self.stabilization_factor = stabilization_factor

    def apply_reinforcement(self, 
                            layer_idx: int, 
                            k: torch.Tensor, 
                            v: torch.Tensor, 
                            stability_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Increases precision (rank or weighting) near successful attractors.
        Suppresses unstable manifold zones.
        """
        # k, v: [batch, heads, seq_len, dim]
        # stability_map: [batch, heads, seq_len]
        
        # Normalize stability map to [0, 1]
        s_map = (stability_map - stability_map.min()) / (stability_map.max() - stability_map.min() + 1e-8)
        
        # Scaling factor: boost stable regions, attenuate unstable ones
        reinforcement_mask = torch.ones_like(s_map) + (s_map - 0.5) * (self.stabilization_factor - 1.0)
        reinforcement_mask = reinforcement_mask.unsqueeze(-1) # [batch, heads, seq_len, 1]
        
        # Apply to V (values are what usually carry the semantic information)
        v_reinforced = v * reinforcement_mask
        
        # 2. Suppression: zero out extremely unstable zones to prevent drift propagation
        unstable_mask = (s_map < 0.1).to(dtype=v.dtype).unsqueeze(-1)
        v_reinforced = v_reinforced * (1.0 - unstable_mask)
        
        return k, v_reinforced

    def allocate_precision_near_attractor(self, 
                                          base_rank: int, 
                                          stability_score: float) -> int:
        """
        Dyamically adjust rank based on local manifold stability.
        """
        if stability_score > 0.8:
            return int(base_rank * 1.5) # Increase precision for stable reasoning cores
        elif stability_score < 0.3:
            return int(base_rank * 0.5) # Shed precision for collapsing zones
        return base_rank

import torch
import numpy as np
from typing import Dict, List

class ResonanceHotspotTracker:
    """
    Tracks resonance intensity and identifies 'hotspots' where cognitive energy is concentrated.
    Used to prioritize stabilization effort.
    """
    def __init__(self, n_heads: int, grid_size: int = 16):
        self.n_heads = n_heads
        self.grid_size = grid_size
        self.hotspot_map = torch.zeros((n_heads, grid_size, grid_size))
        self.resonance_history = []
        
    def update_hotspots(self, resonance_intensities: torch.Tensor, spatial_coords: torch.Tensor):
        """
        resonance_intensities: (n_heads, seq_len)
        spatial_coords: (n_heads, seq_len, 2) normalized to [0, 1]
        """
        # Quantize spatial coords to grid
        coords = (spatial_coords * (self.grid_size - 1)).long()
        
        # Accumulate resonance into grid
        for h in range(self.n_heads):
            for s in range(resonance_intensities.shape[1]):
                x, y = coords[h, s]
                self.hotspot_map[h, x, y] += resonance_intensities[h, s].item()
                
        # Decay old hotspots
        self.hotspot_map *= 0.95
        
    def get_top_hotspots(self, k: int = 5) -> List[Dict]:
        """Returns the most active resonance zones."""
        hotspots = []
        for h in range(self.n_heads):
            val, idx = torch.topk(self.hotspot_map[h].flatten(), k)
            for v, i in zip(val, idx):
                if v > 0.1:
                    x, y = i // self.grid_size, i % self.grid_size
                    hotspots.append({
                        "head": h,
                        "coord": (x.item(), y.item()),
                        "intensity": v.item()
                    })
        return hotspots

    def get_redundancy_score(self) -> float:
        """Calculates how much resonance is concentrated in a few spots."""
        # Entropy of the hotspot map
        probs = self.hotspot_map.flatten() / (self.hotspot_map.sum() + 1e-6)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6)).item()
        return entropy # Low entropy means high redundancy

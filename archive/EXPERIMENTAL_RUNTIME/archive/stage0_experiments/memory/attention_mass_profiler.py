
import torch
import numpy as np
from typing import Dict, List, Optional

class AttentionMassProfiler:
    """
    Phase 20.8: Measures attention mass distribution and fragmentation.
    Identifies 'dilution' in long-context symbolic propagation.
    """
    def __init__(self):
        self.density_history = []
        self.fragmentation_history = []

    def profile_attention(self, attention_weights: torch.Tensor, symbolic_mask: torch.Tensor):
        """
        Calculates density and fragmentation metrics.
        - attention_weights: [kv_len] (averaged across heads/layers)
        - symbolic_mask: [kv_len] (binary mask of symbolic spans)
        """
        weights = attention_weights.detach()
        mask = symbolic_mask.detach()
        
        # 1. Symbolic Density: Total mass on symbolic spans
        symbolic_mass = weights[mask > 0].sum().item()
        total_mass = weights.sum().item()
        density = symbolic_mass / (total_mass + 1e-9)
        
        # 2. Fragmentation: Shannon entropy of the attention distribution
        # High entropy = fragmented attention. Low entropy = concentrated.
        probs = weights / (total_mass + 1e-9)
        fragmentation = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        # 3. Peakiness: Max weight vs average
        peakiness = weights.max().item() / (weights.mean().item() + 1e-9)
        
        entry = {
            "density": density,
            "fragmentation": fragmentation,
            "peakiness": peakiness,
            "symbolic_mass": symbolic_mass
        }
        self.density_history.append(entry)
        return entry

    def get_summary(self):
        if not self.density_history: return {}
        return {
            "avg_density": np.mean([d["density"] for d in self.density_history]),
            "avg_fragmentation": np.mean([d["fragmentation"] for d in self.density_history]),
            "max_peakiness": np.max([d["peakiness"] for d in self.density_history])
        }

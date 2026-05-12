import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class ManifoldDiscoveryEngine:
    """
    Online Manifold Discovery for Phase 34A.
    Detects reusable reasoning regions and prioritizes stable cognitive zones.
    """
    def __init__(self, d_model: int, max_attractors: int = 1000):
        self.d_model = d_model
        self.max_attractors = max_attractors
        self.discovered_manifolds = {} # ID -> {centroid, stability, hits}
        self.active_manifolds = []
        self.stability_threshold = 0.85
        
    def discover_manifolds(self, hidden_states: torch.Tensor, attention_weights: torch.Tensor) -> Dict:
        """
        Processes current hidden states to identify emergent reasoning attractors.
        """
        # Simplify hidden states to manifold centroids
        batch_size, seq_len, _ = hidden_states.shape
        
        # Calculate local stability (inverse of variance in local neighborhood)
        stability = self._estimate_local_stability(hidden_states)
        
        # Identify high-stability regions
        stable_mask = stability > self.stability_threshold
        
        discovered = []
        if stable_mask.any():
            stable_states = hidden_states[stable_mask]
            # Sub-sample to avoid extreme slowdown on CPU
            if len(stable_states) > 32:
                indices = torch.linspace(0, len(stable_states) - 1, 32).long()
                stable_states = stable_states[indices]
                
            for state in stable_states:
                manifold_id = self._match_or_create(state)
                discovered.append(manifold_id)
                
        return {
            "active_ids": list(set(discovered)),
            "stability_map": stability,
            "new_discoveries": len(discovered)
        }

    def _estimate_local_stability(self, states: torch.Tensor) -> torch.Tensor:
        # Measure cosine similarity between adjacent steps as a proxy for stability
        cos_sim = torch.nn.functional.cosine_similarity(states[:, :-1], states[:, 1:], dim=-1)
        # Pad back to original length
        stability = torch.cat([cos_sim, cos_sim[:, -1:]], dim=1)
        return stability

    def _match_or_create(self, state: torch.Tensor) -> str:
        best_id = None
        best_sim = -1.0
        
        for mid, data in self.discovered_manifolds.items():
            sim = torch.nn.functional.cosine_similarity(state.unsqueeze(0), data['centroid'].unsqueeze(0)).item()
            if sim > 0.98: # High similarity threshold for reuse
                if sim > best_sim:
                    best_sim = sim
                    best_id = mid
        
        if best_id:
            self.discovered_manifolds[best_id]['hits'] += 1
            # EMA update for centroid
            self.discovered_manifolds[best_id]['centroid'] = 0.99 * self.discovered_manifolds[best_id]['centroid'] + 0.01 * state
            return best_id
        else:
            new_id = f"manifold_{len(self.discovered_manifolds)}"
            self.discovered_manifolds[new_id] = {
                "centroid": state.clone(),
                "stability": 1.0,
                "hits": 1,
                "birth_step": 0 # Should track global step
            }
            return new_id

    def get_manifold_telemetry(self) -> Dict:
        return {
            "total_manifolds": len(self.discovered_manifolds),
            "reused_count": sum(1 for m in self.discovered_manifolds.values() if m['hits'] > 1),
            "avg_stability": np.mean([m['stability'] for m in self.discovered_manifolds.values()]) if self.discovered_manifolds else 0
        }

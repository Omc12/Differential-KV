import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from .resonance_feedback_engine import ResonanceFeedbackEngine
from .persistent_state_reservoir import PersistentStateReservoir

class RecursiveStabilityController:
    """
    PHASE 25: Recursive Stability Controller
    Coordinates resonance feedback and state persistence to ensure long-range reasoning continuity.
    """
    def __init__(self, 
                 d_model: int, 
                 n_layers: int,
                 feedback_gain: float = 0.15,
                 sync_threshold: float = 0.85):
        self.d_model = d_model
        self.n_layers = n_layers
        self.feedback_gain = feedback_gain
        self.sync_threshold = sync_threshold
        
        self.resonance_engine = ResonanceFeedbackEngine(d_model)
        self.reservoir = PersistentStateReservoir(d_model)
        
        self.layer_coherence = torch.ones(n_layers)
        self.global_resonance_sync = 1.0

    def stabilize_step(self, layer_idx: int, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Applies recursive stabilization to a layer's latent trajectory.
        """
        # 1. Apply Resonance Feedback
        reinforced_state = self.resonance_engine.update_resonance(layer_idx, latent_state)
        
        # 2. Integrate Working Memory Nucleus
        nucleus_signal = self.reservoir.get_nucleus_injection().to(latent_state.device)
        stabilized_state = reinforced_state + self.feedback_gain * nucleus_signal
        
        # 3. Monitor Coherence
        coherence = torch.nn.functional.cosine_similarity(
            latent_state.flatten(), stabilized_state.flatten(), dim=0
        ).item()
        self.layer_coherence[layer_idx] = 0.95 * self.layer_coherence[layer_idx] + 0.05 * coherence
        
        # 4. Synchronize across layers
        self.global_resonance_sync = self.layer_coherence.mean().item()
        
        # 5. Update Reservoir with current stable state
        if layer_idx == self.n_layers - 1: # Last layer
            self.reservoir.update_working_memory_nuclei(stabilized_state)
            
        return stabilized_state

    def check_stability_bounds(self) -> bool:
        """
        Returns True if the system is within stable resonance bounds.
        """
        return self.global_resonance_sync > self.sync_threshold

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "global_resonance_sync": self.global_resonance_sync,
            "min_layer_coherence": self.layer_coherence.min().item(),
            "max_layer_coherence": self.layer_coherence.max().item(),
            "is_stable": self.check_stability_bounds(),
            **self.resonance_engine.get_telemetry()
        }

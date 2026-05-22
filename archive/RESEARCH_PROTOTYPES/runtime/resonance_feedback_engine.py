import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ResonanceState:
    attractor_base: torch.Tensor
    phase_velocity: torch.Tensor
    coherence_score: float
    last_reinforcement_step: int

class ResonanceFeedbackEngine:
    """
    PHASE 25: Recursive Cognitive Resonance (RCR)
    Implements recursive latent reinforcement to prevent reasoning collapse at ultra-long horizons.
    """
    def __init__(self, 
                 d_model: int, 
                 reinforcement_interval: int = 8,
                 reinforcement_gain: float = 0.15,
                 decay_suppression: float = 0.999):
        self.d_model = d_model
        self.reinforcement_interval = reinforcement_interval
        self.reinforcement_gain = reinforcement_gain
        self.decay_suppression = decay_suppression
        
        self.active_resonances: Dict[int, ResonanceState] = {}
        self.step_counter = 0
        
    def update_resonance(self, layer_idx: int, latent_state: torch.Tensor):
        """
        Updates and reinforces the resonance state for a given layer.
        """
        self.step_counter += 1
        
        if layer_idx not in self.active_resonances:
            self.active_resonances[layer_idx] = ResonanceState(
                attractor_base=latent_state.detach().clone(),
                phase_velocity=torch.zeros_like(latent_state),
                coherence_score=1.0,
                last_reinforcement_step=self.step_counter
            )
            return latent_state

        state = self.active_resonances[layer_idx]
        
        # Calculate current coherence (cosine similarity to base)
        current_coherence = torch.nn.functional.cosine_similarity(
            latent_state.flatten(), state.attractor_base.flatten(), dim=0
        ).item()
        
        # Update phase velocity
        delta = latent_state - state.attractor_base
        state.phase_velocity = 0.9 * state.phase_velocity + 0.1 * delta
        
        # Adaptive reinforcement scheduling
        needs_reinforcement = (self.step_counter - state.last_reinforcement_step) >= self.reinforcement_interval
        
        if needs_reinforcement and current_coherence > 0.7:
            # Recursive Latent Reinforcement
            # Reinforce the coherent attractor to push back against decay
            latent_state = latent_state + self.reinforcement_gain * state.attractor_base
            
            # Maintain phase continuity by projecting forward
            latent_state = latent_state + state.phase_velocity * 0.1
            
            # Update state base with new reinforced position
            state.attractor_base = 0.95 * state.attractor_base + 0.05 * latent_state.detach()
            state.last_reinforcement_step = self.step_counter
            state.coherence_score = current_coherence
        else:
            # Suppress resonance decay if not reinforcing
            state.coherence_score *= self.decay_suppression
            
        return latent_state

    def get_telemetry(self) -> Dict[str, Any]:
        if not self.active_resonances:
            return {}
            
        avg_coherence = np.mean([s.coherence_score for s in self.active_resonances.values()])
        return {
            "avg_resonance_coherence": avg_coherence,
            "active_attractors": len(self.active_resonances),
            "step_count": self.step_counter
        }

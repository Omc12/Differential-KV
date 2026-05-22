"""
runtime/resonance_pulse_controller.py
Phase 26: Cognitive Energy Minimization (CEM)
Controller to trigger and manage resonance pulses.
"""

import torch
from typing import List, Optional, Dict
from runtime.resonance_feedback_engine import ResonanceFeedbackEngine

class ResonancePulseController:
    def __init__(self, resonance_engine: ResonanceFeedbackEngine):
        self.resonance_engine = resonance_engine
        self.total_pulses = 0

    def apply_pulse(self, layer_idx: int, latent_state: torch.Tensor, intensity: float = 1.5) -> torch.Tensor:
        """
        Applies a localized resonance pulse to stabilize a layer's trajectory.
        Higher intensity than standard Phase 25 reinforcement to compensate for sparsity.
        """
        self.total_pulses += 1
        
        if layer_idx not in self.resonance_engine.active_resonances:
            # Initialize resonance if not present
            return self.resonance_engine.update_resonance(layer_idx, latent_state)

        state = self.resonance_engine.active_resonances[layer_idx]
        
        # Calculate pulse gain: engine's base gain scaled by intensity
        pulse_gain = self.resonance_engine.reinforcement_gain * intensity
        
        # 1. Attractor Reinforcement
        # Directly push the latent state back towards the stable attractor base
        reinforcement = pulse_gain * state.attractor_base
        latent_state = latent_state + reinforcement
        
        # 2. Phase Correction
        # Adjust for velocity to maintain temporal continuity
        phase_correction = state.phase_velocity * 0.15 
        latent_state = latent_state + phase_correction
        
        # 3. State Update
        # Update the attractor base with the new reinforced position
        # We use a slightly higher update rate for pulses to quickly lock in the stabilization
        state.attractor_base = 0.9 * state.attractor_base + 0.1 * latent_state.detach()
        state.last_reinforcement_step = self.resonance_engine.step_counter
        
        return latent_state

    def get_telemetry(self) -> Dict:
        return {
            "total_pulses": self.total_pulses,
            "resonance_engine_status": self.resonance_engine.get_telemetry()
        }

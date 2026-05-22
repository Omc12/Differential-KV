import torch
import numpy as np
from typing import Dict
from regulation.synchronization_damping_engine import SynchronizationDampingEngine
from regulation.adaptive_phase_release import AdaptivePhaseRelease

class ResonancePressureController:
    """
    Prevents resonance lock spirals by balancing pressure and damping synchronization.
    Implements anti-lock mechanisms for distributed coherence.
    """
    def __init__(self, pressure_threshold: float = 2.0):
        self.threshold = pressure_threshold
        self.damping_engine = SynchronizationDampingEngine()
        self.phase_release = AdaptivePhaseRelease()
        
    def control_pressure(self, manifold_states: torch.Tensor, current_pressure: float) -> torch.Tensor:
        """
        Adjusts the manifold states to release pressure if it exceeds thresholds.
        """
        if current_pressure > self.threshold:
            # High pressure detected - apply damping or release
            if current_pressure > self.threshold * 2:
                # Emergency desynchronization to prevent lock spiral
                manifold_states = self.phase_release.apply_release(manifold_states)
            else:
                # Moderate damping
                manifold_states = self.damping_engine.apply_damping(manifold_states, current_pressure)
                
        return manifold_states
        
    def get_controller_state(self) -> Dict:
        return {
            "is_damping": self.damping_engine.active,
            "is_releasing": self.phase_release.active,
            "damping_intensity": self.damping_engine.intensity,
            "release_count": self.phase_release.release_history_count
        }

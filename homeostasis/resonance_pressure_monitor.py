import torch
import numpy as np
from typing import Dict

class ResonancePressureMonitor:
    """
    Monitors 'resonance pressure' - the internal tension within the manifold 
    caused by high synchronization density or attractor volatility.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.state_history = []
        self.pressure_history = []
        
    def update(self, manifold_states: torch.Tensor) -> Dict:
        """
        Updates pressure metrics based on new manifold states.
        """
        # Calculate manifold velocity (rate of change)
        current_state = manifold_states.detach().cpu().numpy()
        
        velocity = 0.0
        gradient = 0.0
        
        if self.state_history:
            prev_state = self.state_history[-1]
            velocity = np.linalg.norm(current_state - prev_state)
            
            if len(self.state_history) > 1:
                # Gradient of velocity (acceleration)
                prev_velocity = np.linalg.norm(prev_state - self.state_history[-2])
                gradient = abs(velocity - prev_velocity)
        
        # Manifold pressure is a function of velocity and internal coherence
        # High velocity + high incoherence = high pressure
        manifold_pressure = velocity * (1.0 + gradient)
        
        self.state_history.append(current_state)
        self.pressure_history.append(manifold_pressure)
        
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
            self.pressure_history.pop(0)
            
        return {
            "manifold_pressure": manifold_pressure,
            "pressure_gradient": gradient,
            "is_volatile": gradient > 0.8
        }
        
    def get_saturation_risk(self) -> float:
        """
        Predicts if the manifold is approaching a 'resonance lock' saturation state.
        """
        if not self.pressure_history:
            return 0.0
        return float(np.mean(self.pressure_history) / 10.0) # Normalized risk

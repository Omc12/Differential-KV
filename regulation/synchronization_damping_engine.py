import torch

class SynchronizationDampingEngine:
    """
    Applies anti-lock synchronization damping to prevent manifold rigidity.
    """
    def __init__(self):
        self.active = False
        self.intensity = 0.0
        
    def apply_damping(self, states: torch.Tensor, pressure: float) -> torch.Tensor:
        """
        Smoothly reduces resonance effects based on pressure.
        """
        self.active = True
        # Calculate damping intensity proportional to pressure overflow
        self.intensity = min(0.9, (pressure - 1.0) / 5.0)
        
        # Damping works by adding a small amount of 'noise' or 'jitter' 
        # to break rigid synchronization patterns.
        jitter = torch.randn_like(states) * 0.01 * self.intensity
        return states * (1.0 - self.intensity * 0.1) + jitter
        
    def reset(self):
        self.active = False
        self.intensity = 0.0
        

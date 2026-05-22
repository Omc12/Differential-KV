import torch

class SparseStabilityController:
    """
    Maintains sparse attention stability at extreme context lengths (512k+).
    Suppresses 'jitter' caused by rapid changes in sparse masks.
    """
    def __init__(self, damping_factor: float = 0.5):
        self.damping_factor = damping_factor
        self.prev_mask = None

    def stabilize_mask(self, current_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies temporal damping to the sparse mask to prevent oscillation.
        """
        if self.prev_mask is None:
            self.prev_mask = current_mask
            return current_mask
            
        # A token is preserved if it was in the previous mask OR it's in the current mask
        # and satisfies a persistence condition (simulated here)
        
        # In a real system, we'd use a more sophisticated hysteresis or momentum approach
        damped_mask = current_mask | (self.prev_mask & (torch.rand_like(current_mask.float()) > self.damping_factor))
        
        self.prev_mask = damped_mask
        return damped_mask

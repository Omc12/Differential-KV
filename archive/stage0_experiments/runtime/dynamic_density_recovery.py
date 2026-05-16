import torch

class DynamicDensityRecovery:
    """
    Triggers rapid density recovery when stability is compromised.
    Briefly increases KV retention to re-stabilize attention distributions.
    """
    def __init__(self, baseline_density: float = 0.1):
        self.baseline_density = baseline_density
        self.recovery_mode = False
        self.recovery_steps = 0

    def get_density(self, collapse_imminent: bool):
        """
        Calculates the required density for the current step.
        """
        if collapse_imminent:
            self.recovery_mode = True
            self.recovery_steps = 10 # Stay in recovery for 10 steps
            
        if self.recovery_mode:
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_mode = False
            return min(1.0, self.baseline_density * 3) # Triple the density
            
        return self.baseline_density

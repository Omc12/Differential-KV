import time
from typing import Literal

class AnchorModeTransitionController:
    """
    PHASE 7.5A: Anchor Mode Transition Controller
    Manages the transition between different anchor strategies (e.g., STATIC, DYNAMIC, HYBRID)
    to prevent performance spikes or retrieval collapses during runtime changes.
    """
    def __init__(self):
        self.current_mode: Literal["STATIC", "DYNAMIC", "HYBRID"] = "STATIC"
        self.transition_start_time = 0
        self.transition_duration = 5.0 # Seconds
        self.is_transitioning = False

    def request_transition(self, target_mode: str):
        """Initiates a gradual transition to a new mode."""
        if target_mode == self.current_mode:
            return
            
        self.target_mode = target_mode
        self.transition_start_time = time.time()
        self.is_transitioning = True

    def get_interpolation_factor(self) -> float:
        """
        Calculates the weight of the target mode (0.0 to 1.0).
        """
        if not self.is_transitioning:
            return 1.0 if self.current_mode == "STATIC" else 0.0 # Simplified
            
        elapsed = time.time() - self.transition_start_time
        factor = min(1.0, elapsed / self.transition_duration)
        
        if factor >= 1.0:
            self.current_mode = self.target_mode
            self.is_transitioning = False
            
        return factor

    def get_active_mode(self) -> str:
        """Returns the mode that should be predominantly used."""
        return self.current_mode if not self.is_transitioning else f"TRANSITIONING_TO_{self.target_mode}"

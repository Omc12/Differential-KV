
import time
from typing import Dict

class RecallDecayController:
    """
    PHASE 21.1: Manages temporary symbolic reinforcement and recall fading.
    Prevents symbolic fixation loops (where the model gets stuck in a recall).
    """
    def __init__(self, decay_rate: float = 0.9, min_strength: float = 0.1):
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        self.active_strengths: Dict[str, float] = {}

    def update_reinforcement(self, hub_id: str, initial_strength: float):
        """Initializes or resets reinforcement for a hub."""
        self.active_strengths[hub_id] = initial_strength

    def apply_decay(self, hub_id: str) -> float:
        """Calculates current strength after decay."""
        strength = self.active_strengths.get(hub_id, 0.0)
        
        # Decay the strength
        new_strength = strength * self.decay_rate
        
        if new_strength < self.min_strength:
            new_strength = 0.0
            if hub_id in self.active_strengths:
                del self.active_strengths[hub_id]
        else:
            self.active_strengths[hub_id] = new_strength
            
        return new_strength

    def reset_decay(self, hub_id: str):
        if hub_id in self.active_strengths:
            del self.active_strengths[hub_id]

    def is_active(self, hub_id: str) -> bool:
        return hub_id in self.active_strengths and self.active_strengths[hub_id] > 0

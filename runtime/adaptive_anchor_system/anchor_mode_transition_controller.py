import torch
from typing import Dict, List
from runtime.adaptive_anchor_system.adaptive_anchor_modes import AnchorSpacingMode

class AnchorModeTransitionController:
    """
    Ensures gradual and smooth transitions between anchor spacing modes 
    to prevent performance jitter and anchor oscillation.
    """
    def __init__(self, hysteresis: float = 0.1, cooldown_steps: int = 3):
        self.hysteresis = hysteresis
        self.cooldown_steps = cooldown_steps
        self.current_modes: Dict[int, AnchorSpacingMode] = {}
        self.cooldown_counters: Dict[int, int] = {}

    def get_transitioned_mode(self, 
                              chunk_idx: int, 
                              target_mode: AnchorSpacingMode) -> AnchorSpacingMode:
        """
        Calculates the next mode for a chunk, applying hysteresis and cooldown.
        """
        current_mode = self.current_modes.get(chunk_idx, AnchorSpacingMode.SPARSE)
        
        # If already in cooldown, stay in current mode
        if self.cooldown_counters.get(chunk_idx, 0) > 0:
            self.cooldown_counters[chunk_idx] -= 1
            return current_mode
            
        # Only transition if target is more than 1 level away or after a delay
        # Smaller IntEnum value = Denser spacing
        if target_mode < current_mode: # Stepping up density (e.g. 512 -> 256)
            # Immediate transition for safety if retrieval is failing
            self.current_modes[chunk_idx] = target_mode
            self.cooldown_counters[chunk_idx] = self.cooldown_steps
            return target_mode
        elif target_mode > current_mode: # Stepping down density (e.g. 128 -> 256)
            # Gradual transition: only step down one level at a time
            new_mode = self._get_next_sparse_mode(current_mode)
            self.current_modes[chunk_idx] = new_mode
            self.cooldown_counters[chunk_idx] = self.cooldown_steps
            return new_mode
            
        return current_mode

    def _get_next_sparse_mode(self, current: AnchorSpacingMode) -> AnchorSpacingMode:
        """Helper to move one step towards SPARSER spacing."""
        modes = sorted([m for m in AnchorSpacingMode], reverse=False) # 64, 128, 256, 512
        try:
            idx = modes.index(current)
            if idx < len(modes) - 1:
                return modes[idx + 1]
        except ValueError:
            pass
        return current

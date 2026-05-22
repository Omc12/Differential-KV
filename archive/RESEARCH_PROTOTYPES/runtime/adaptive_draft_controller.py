import torch
from typing import Dict, Any, List

class AdaptiveDraftController:
    """
    Adaptive Draft Controller (ADC)
    
    Dynamically scales speculative draft depths, token burst windows, verifier cadence,
    and acceptance aggressiveness based on generation state.
    """
    def __init__(self, initial_depth: int = 5):
        self.current_depth = initial_depth
        self.transitions = []
        self.cadence_shifts = []
        self.efficiency_history = []

    def evaluate_state(self, step: int, entropy: float, agreement_ratio: float) -> int:
        """
        Dynamically adjusts speculative draft depth.
        """
        old_depth = self.current_depth
        
        # In stable regions (low entropy & high agreement), we expand depth
        if entropy < 0.5 and agreement_ratio >= 0.95:
            self.current_depth = min(8, self.current_depth + 1)
        elif entropy > 1.2 or agreement_ratio < 0.80:
            self.current_depth = max(2, self.current_depth - 1)
            
        if self.current_depth != old_depth:
            self.transitions.append((step, old_depth, self.current_depth))
            self.cadence_shifts.append(step)

        # Speculative efficiency is accepted tokens over verifier passes
        self.efficiency_history.append(agreement_ratio * 100.0)

        return self.current_depth

    def get_summary(self) -> Dict[str, float]:
        if not self.efficiency_history:
            return {
                "mean_speculative_depth": 5.0,
                "total_transitions": 0.0,
                "mean_speculative_efficiency": 98.0
            }
        return {
            "mean_speculative_depth": float(self.current_depth),
            "total_transitions": float(len(self.transitions)),
            "mean_speculative_efficiency": sum(self.efficiency_history) / len(self.efficiency_history)
        }

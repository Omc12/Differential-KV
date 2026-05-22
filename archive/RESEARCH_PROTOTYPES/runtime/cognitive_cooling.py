"""
runtime/cognitive_cooling.py
Phase 26: Cognitive Energy Minimization (CEM)
Dynamically reduces repair intensity and synchronization during low-risk reasoning regions.
"""

from enum import Enum
from typing import Dict, List

class CognitionMode(Enum):
    HOT = 1   # High risk, high energy: Max intervention and frequent synchronization
    WARM = 2  # Moderate risk: Balanced intervention and periodic synchronization
    COOL = 3  # Low risk, low energy: Minimal intervention, self-sustaining manifolds

class CognitiveCoolingScheduler:
    def __init__(self):
        self.current_mode = CognitionMode.WARM
        self.mode_history = []
        self.energy_history = []

    def update_mode(self, energy: float, stability_score: float):
        """
        Determines the cognition mode based on current latent energy and stability.
        HOT: Needs active repair.
        COOL: Trust the passive stability basins.
        """
        self.energy_history.append(energy)
        
        # Mode transition logic
        if energy > 0.4 or stability_score < 0.6:
            self.current_mode = CognitionMode.HOT
        elif energy < 0.1 and stability_score > 0.9:
            # We only cool down if we've been stable for a few steps
            if len(self.energy_history) > 3 and all(e < 0.15 for e in self.energy_history[-3:]):
                self.current_mode = CognitionMode.COOL
        else:
            self.current_mode = CognitionMode.WARM
            
        self.mode_history.append(self.current_mode)

    def get_repair_intensity(self) -> float:
        """Scales the magnitude of stabilization interventions."""
        if self.current_mode == CognitionMode.HOT:
            return 1.0
        elif self.current_mode == CognitionMode.WARM:
            return 0.4
        else: # COOL
            return 0.05 # Near-zero intervention

    def get_sync_frequency(self) -> int:
        """Determines how often cross-layer synchronization should occur."""
        if self.current_mode == CognitionMode.HOT:
            return 1 # Every step
        elif self.current_mode == CognitionMode.WARM:
            return 4 # Every 4 steps
        else: # COOL
            return 16 # Rare sync

    def get_telemetry(self) -> Dict:
        return {
            "current_mode": self.current_mode.name,
            "repair_intensity": self.get_repair_intensity(),
            "sync_frequency": self.get_sync_frequency(),
            "mode_transition_count": sum(1 for i in range(1, len(self.mode_history)) if self.mode_history[i] != self.mode_history[i-1])
        }

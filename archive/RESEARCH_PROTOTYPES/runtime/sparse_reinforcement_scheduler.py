"""
runtime/sparse_reinforcement_scheduler.py
Phase 26: Cognitive Energy Minimization (CEM)
Triggers resonance pulses only when necessary based on energy/coherence spikes.
"""

import numpy as np
from typing import Dict, List, Optional

class SparseReinforcementScheduler:
    def __init__(self, energy_threshold: float = 0.3, coherence_decay_threshold: float = 0.05):
        self.energy_threshold = energy_threshold
        self.coherence_decay_threshold = coherence_decay_threshold
        self.prev_coherence = 1.0
        self.pulse_history = []
        self.step_count = 0

    def should_trigger_pulse(self, energy: float, coherence: float, desync: float) -> bool:
        """
        Evaluates cognitive metrics to decide if a resonance pulse is required.
        """
        self.step_count += 1
        coherence_decay = self.prev_coherence - coherence
        # Small decay is normal, we care about acceleration in decay
        self.prev_coherence = coherence
        
        trigger = False
        
        # Condition 1: Energy Spike
        # High energy indicates the trajectory is entering an unstable or collapse basin.
        if energy > self.energy_threshold:
            trigger = True
        
        # Condition 2: Coherence Decay
        # Rapid loss of alignment with the base attractor.
        if coherence_decay > self.coherence_decay_threshold:
            trigger = True
            
        # Condition 3: Phase Divergence
        # Layers are drifting apart in their geometric timing.
        if desync > 0.4:
            trigger = True

        if trigger:
            self.pulse_history.append(self.step_count)
            
        return trigger

    def get_pulse_frequency(self) -> float:
        """Returns the ratio of pulses to total reasoning steps."""
        if self.step_count == 0: return 0.0
        return len(self.pulse_history) / self.step_count
    
    def get_telemetry(self) -> Dict:
        return {
            "pulse_count": len(self.pulse_history),
            "pulse_frequency": self.get_pulse_frequency(),
            "last_pulse_step": self.pulse_history[-1] if self.pulse_history else 0
        }

"""
runtime/efficiency_controller.py
Phase 26: Cognitive Energy Minimization (CEM)
Optimizes coherence / FLOP by balancing reasoning quality and intervention density.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from analysis.resonance_energy import CognitiveEnergyModel
from runtime.cognitive_cooling import CognitiveCoolingScheduler
from runtime.sparse_reinforcement_scheduler import SparseReinforcementScheduler
from runtime.resonance_pulse_controller import ResonancePulseController

class EfficiencyAwareRuntimeController:
    """
    The orchestrator for Phase 26. 
    It integrates energy modeling, cooling schedules, and sparse pulses to 
    minimize runtime overhead while maintaining reasoning stability.
    """
    def __init__(self, 
                 resonance_engine, 
                 energy_model: Optional[CognitiveEnergyModel] = None,
                 cooling_scheduler: Optional[CognitiveCoolingScheduler] = None,
                 pulse_scheduler: Optional[SparseReinforcementScheduler] = None,
                 pulse_controller: Optional[ResonancePulseController] = None):
        self.resonance_engine = resonance_engine
        self.energy_model = energy_model or CognitiveEnergyModel()
        self.cooling_scheduler = cooling_scheduler or CognitiveCoolingScheduler()
        self.pulse_scheduler = pulse_scheduler or SparseReinforcementScheduler()
        self.pulse_controller = pulse_controller or ResonancePulseController(resonance_engine)
        
        self.step_counter = 0
        self.efficiency_history = []

    def process_layer(self, 
                      layer_idx: int, 
                      latent_state: torch.Tensor, 
                      metrics: Dict[str, float]) -> torch.Tensor:
        """
        Processes a single layer through the efficiency-aware pipeline.
        Determines if an active intervention (pulse) is needed or if 
        the trajectory can remain in a passive stability basin.
        """
        self.step_counter += 1
        
        # 1. Energy Analysis
        energy = self.energy_model.record_energy(metrics)
        coherence = metrics.get("cognitive_stability_score", 1.0)
        desync = metrics.get("phase_desync", 0.0)
        
        # 2. Update Cooling Mode
        # This adjusts global parameters like repair intensity and sync frequency
        self.cooling_scheduler.update_mode(energy, coherence)
        repair_intensity = self.cooling_scheduler.get_repair_intensity()
        
        # 3. Sparse Reinforcement Pulse Logic
        # Instead of continuous Phase 25 reinforcement, we trigger sparse pulses.
        should_pulse = self.pulse_scheduler.should_trigger_pulse(energy, coherence, desync)
        
        if should_pulse:
            # Apply active repair with intensity scaled by the cooling scheduler
            latent_state = self.pulse_controller.apply_pulse(
                layer_idx, 
                latent_state, 
                intensity=repair_intensity * 2.0 # Higher intensity for sparse pulses
            )
        else:
            # Passive Stability Basin
            # We don't apply reinforcement here, saving computation (FLOPs).
            # We still update the engine's internal step counter to keep sync.
            self.resonance_engine.step_counter += 1
            
        # 4. Efficiency Tracking
        # Metric: Coherence / (Intervention Density + 1)
        intervention_density = self.pulse_scheduler.get_pulse_frequency()
        efficiency = coherence / (intervention_density + 0.1)
        self.efficiency_history.append(efficiency)
        
        return latent_state

    def get_telemetry(self) -> Dict[str, Any]:
        """Aggregates telemetry from all sub-modules."""
        return {
            "step_count": self.step_counter,
            "current_energy": self.energy_model.get_history()[-1] if self.energy_model.get_history() else 0,
            "cooling_mode": self.cooling_scheduler.current_mode.name,
            "pulse_frequency": self.pulse_scheduler.get_pulse_frequency(),
            "avg_efficiency": float(np.mean(self.efficiency_history)) if self.efficiency_history else 0.0,
            "sync_required": (self.step_counter % self.cooling_scheduler.get_sync_frequency() == 0)
        }

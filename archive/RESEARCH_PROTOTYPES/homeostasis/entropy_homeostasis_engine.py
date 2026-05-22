import torch
import numpy as np
from typing import Dict, List, Optional
from homeostasis.global_entropy_regulator import GlobalEntropyRegulator
from homeostasis.resonance_pressure_monitor import ResonancePressureMonitor

class EntropyHomeostasisEngine:
    """
    Main engine for maintaining cognitive equilibrium through continuous entropy regulation.
    Integrates pressure monitoring and global regulation to prevent entropy explosions.
    """
    def __init__(self, d_model: int, window_size: int = 100):
        self.d_model = d_model
        self.regulator = GlobalEntropyRegulator(d_model)
        self.pressure_monitor = ResonancePressureMonitor(window_size)
        
        self.entropy_trajectory = []
        self.pressure_trajectory = []
        self.equilibrium_state = True
        
    def maintain_homeostasis(self, manifold_states: torch.Tensor) -> Dict:
        """
        Performs one step of homeostasis maintenance.
        """
        # 1. Monitor current state
        entropy_stats = self.regulator.measure_entropy(manifold_states)
        pressure_stats = self.pressure_monitor.update(manifold_states)
        
        # 2. Check for equilibrium drift
        is_drifting = self.regulator.detect_drift(entropy_stats['current_entropy'])
        
        # 3. Apply stabilization if needed
        correction_factor = 1.0
        if is_drifting or pressure_stats['pressure_gradient'] > 0.5:
            correction_factor = self.regulator.calculate_correction(
                entropy_stats['current_entropy'],
                pressure_stats['manifold_pressure']
            )
            self.equilibrium_state = False
        else:
            self.equilibrium_state = True
            
        # 4. Adaptive cooling if entropy is low
        if entropy_stats['current_entropy'] < self.regulator.target_entropy:
            self.regulator.apply_cooling(0.05)
            
        # Record trajectories
        self.entropy_trajectory.append(entropy_stats['current_entropy'])
        self.pressure_trajectory.append(pressure_stats['manifold_pressure'])
        
        return {
            "entropy": entropy_stats['current_entropy'],
            "pressure": pressure_stats['manifold_pressure'],
            "correction_factor": correction_factor,
            "is_stable": self.equilibrium_state,
            "cooling_active": self.regulator.cooling_intensity > 0
        }

    def get_ecosystem_health(self) -> Dict:
        """
        Returns a summary of the cognitive ecosystem health.
        """
        return {
            "entropy_volatility": np.std(self.entropy_trajectory[-50:]) if len(self.entropy_trajectory) > 1 else 0,
            "pressure_density": np.mean(self.pressure_trajectory[-50:]) if len(self.pressure_trajectory) > 1 else 0,
            "equilibrium_status": "STABLE" if self.equilibrium_state else "CORRECTING",
            "long_horizon_drift": self.regulator.estimate_long_horizon_drift(self.entropy_trajectory)
        }

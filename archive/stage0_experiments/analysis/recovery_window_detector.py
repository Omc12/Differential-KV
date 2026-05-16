"""
analysis/recovery_window_detector.py

Detects the 'Recovery Window' - the optimal temporal interval for intervention
before a reasoning trajectory enters an irreversible collapse basin.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class RecoveryWindowDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.collapse_history = []
        self.critical_threshold = self.config.get("critical_threshold", 0.6)

    def analyze_recovery_potential(self, 
                                   health_history: List[float], 
                                   drift_history: List[float]) -> Dict[str, Any]:
        """
        Predicts the optimal repair step and the width of the recovery window.
        """
        if len(health_history) < 3:
            return {"optimal_repair_step": -1, "recovery_window_width": 0}

        # Calculate acceleration of collapse (second derivative of health)
        health_arr = np.array(health_history)
        health_velocity = np.diff(health_arr)
        health_acceleration = np.diff(health_velocity)
        
        # Calculate drift acceleration
        drift_arr = np.array(drift_history)
        drift_velocity = np.diff(drift_arr)

        # 1. Pre-collapse signal detection
        # Signals: Health dropping, acceleration turning negative, drift increasing
        is_failing = health_velocity[-1] < 0 and health_arr[-1] < 0.9
        
        # 2. Recovery Horizon Estimation
        # How many steps until we hit the 'point of no return' (e.g. health < 0.2)
        if health_velocity[-1] < 0:
            steps_to_failure = (health_arr[-1] - 0.2) / abs(health_velocity[-1])
        else:
            steps_to_failure = 100 # Stable
            
        # 3. Optimal Repair Step
        # Ideally, we repair just as acceleration peaks (the 'bend' in the curve)
        # or when health drops below a tolerance but before it hits the cliff.
        current_step = len(health_history)
        if is_failing:
            # Optimal was likely a few steps ago or right now
            optimal_repair_step = max(0, current_step - 2)
        else:
            optimal_repair_step = current_step

        # 4. Recovery Window Width
        # The time between first signal and irreversibility
        # Width = (Step where health < 0.8) to (Step where health < 0.3)
        failure_indices = np.where(health_arr < 0.8)[0]
        critical_indices = np.where(health_arr < 0.3)[0]
        
        if len(failure_indices) > 0 and len(critical_indices) > 0:
            window_width = critical_indices[0] - failure_indices[0]
        else:
            window_width = steps_to_failure * 0.5 # Estimate based on velocity

        return {
            "optimal_repair_step": int(optimal_repair_step),
            "predicted_time_to_failure": float(steps_to_failure),
            "recovery_window_width": float(window_width),
            "is_critical": bool(health_arr[-1] < self.critical_threshold),
            "collapse_velocity": float(health_velocity[-1]),
            "divergence_forecast": float(drift_velocity[-1] * steps_to_failure if len(drift_velocity) > 0 else 0)
        }

    def forecast_trajectory_divergence(self, latent_states: torch.Tensor, steps_ahead: int = 5) -> torch.Tensor:
        """
        Extrapolates the latent trajectory to predict future divergence.
        """
        # Simple linear extrapolation of the last two states for the prototype
        if latent_states.shape[0] < 2:
            return latent_states[-1].repeat(steps_ahead, 1)
        
        v = latent_states[-1] - latent_states[-2]
        forecast = []
        for i in range(1, steps_ahead + 1):
            forecast.append(latent_states[-1] + v * i)
        
        return torch.stack(forecast)

"""
STAGE 2 - ASS: Predictive Anchor Stability Analyzer
Phase 39.5 - Adaptive Semantic Scheduling

Predicts anchor weakening BEFORE semantic degradation occurs.
"""
import threading
from typing import Dict, Any, List

class PredictiveAnchorStabilityAnalyzer:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        self._anchor_age: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self._predicted_failure_step: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self._half_life: Dict[int, float] = {i: 20.0 for i in range(num_layers)} # Initial guess

    def record_step(self, step: int, layer_idx: int, is_dense: bool, drift_delta: float):
        with self._lock:
            if is_dense:
                # Reinforcement
                # If it was decaying quickly, update half-life
                if self._anchor_age[layer_idx] > 0 and drift_delta > 0:
                     # Shorter life if drift grew fast
                     decay_rate = drift_delta / max(self._anchor_age[layer_idx], 1)
                     new_half_life = max(5.0, 0.1 / max(decay_rate, 0.001))
                     # Smooth update
                     self._half_life[layer_idx] = (self._half_life[layer_idx] * 0.8) + (new_half_life * 0.2)
                
                self._anchor_age[layer_idx] = 0
                self._predicted_failure_step[layer_idx] = step + int(self._half_life[layer_idx])
            else:
                self._anchor_age[layer_idx] += 1

    def will_fail_soon(self, step: int, layer_idx: int, lookahead: int = 5) -> bool:
        with self._lock:
            return (step + lookahead) >= self._predicted_failure_step.get(layer_idx, step + 100)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_half_life = sum(self._half_life.values()) / max(self.num_layers, 1)
            return {
                "avg_anchor_half_life": round(avg_half_life, 2)
            }

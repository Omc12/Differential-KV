"""
STAGE 2 - ASS: Predictive Semantic Pressure Estimator
Phase 39.5 - Adaptive Semantic Scheduling

Estimates the probability of future semantic instability BEFORE drift spikes occur.
Uses a combination of momentum, continuity decay, and anchor degradation.
"""
import threading
from typing import Dict, Any, List

class PredictiveSemanticPressureEstimator:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        # Historical tracking for velocity calculations
        self._drift_history: Dict[int, List[float]] = {i: [] for i in range(num_layers)}
        self._repair_history: Dict[int, List[int]] = {i: [] for i in range(num_layers)}
        self._history_len = 10
        
        self._current_pressure: Dict[int, float] = {i: 0.0 for i in range(num_layers)}

    def record_state(self, step: int, layer_idx: int, drift: float, repaired: bool, anchor_age: int, continuity_chain: int):
        with self._lock:
            # 1. Update histories
            self._drift_history[layer_idx].append(drift)
            if len(self._drift_history[layer_idx]) > self._history_len:
                self._drift_history[layer_idx].pop(0)
                
            self._repair_history[layer_idx].append(1 if repaired else 0)
            if len(self._repair_history[layer_idx]) > self._history_len:
                self._repair_history[layer_idx].pop(0)
            
            # 2. Calculate components
            drift_velocity = self._calculate_velocity(self._drift_history[layer_idx])
            repair_frequency = sum(self._repair_history[layer_idx]) / max(len(self._repair_history[layer_idx]), 1)
            
            # Decay factor: longer chains reduce pressure, but if drift velocity is high, it overrides
            continuity_decay = max(0.0, 1.0 - (continuity_chain / 50.0))
            
            # Anchor strain increases pressure if age is high
            anchor_strain = min(anchor_age / 20.0, 1.0)
            
            # 3. Compute final pressure score (0.0 to 1.0)
            # Pressure builds up if drift is accelerating, repairs are frequent, or anchors are old
            base_pressure = (drift_velocity * 0.4) + (repair_frequency * 0.3) + (anchor_strain * 0.3)
            adjusted_pressure = base_pressure * (0.5 + 0.5 * continuity_decay)
            
            self._current_pressure[layer_idx] = max(0.0, min(1.0, adjusted_pressure))

    def _calculate_velocity(self, history: List[float]) -> float:
        if len(history) < 2: return 0.0
        # Simple difference between most recent and average of previous
        recent = history[-1]
        past_avg = sum(history[:-1]) / len(history[:-1])
        velocity = recent - past_avg
        return max(0.0, velocity) # We only care about positive velocity (increasing drift)

    def get_pressure(self, layer_idx: int) -> float:
        with self._lock:
            return self._current_pressure.get(layer_idx, 0.0)

    def get_global_pressure(self) -> float:
        with self._lock:
            return sum(self._current_pressure.values()) / max(self.num_layers, 1)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_pressure = self.get_global_pressure()
            critical_layers = sum(1 for p in self._current_pressure.values() if p > 0.7)
            return {
                "avg_semantic_pressure": round(avg_pressure, 4),
                "high_pressure_layers": critical_layers
            }

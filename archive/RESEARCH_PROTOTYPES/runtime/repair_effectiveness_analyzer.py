"""
STAGE 2 - SDR: Repair Effectiveness Optimizer
Phase 39.4 - Semantic Drift Reduction

Expanded analyzer to measure stabilization persistence and semantic retention.
Repairs only count as SUCCESSFUL if semantic drift materially decreases afterward.
"""
import threading
import time
from typing import Dict, Any, List, Deque
from collections import deque


class RepairEffectivenessAnalyzer:
    """
    Tracks whether semantic repair activations actually reduced KL divergence
    and how long that stabilization lasts (persistence).
    """
    EFFECTIVENESS_DELTA = 0.02  # Minimum required drift reduction
    
    def __init__(self, history_len: int = 20):
        self._lock = threading.RLock()
        self._activations = 0
        self._effective = 0
        self._ineffective = 0
        self._total_drift_reduction = 0.0
        
        # persistence tracking: layer -> list of steps since last repair
        self._layer_persistence: Dict[int, List[int]] = {}
        self._last_repair_step: Dict[int, int] = {}
        
        # drift history for stabilization half-life: layer -> deque[float]
        self._drift_history: Dict[int, Deque[float]] = {}
        self.history_len = history_len

        # layer-level stats
        self._layer_activations: Dict[int, int] = {}
        self._layer_effective: Dict[int, int] = {}
        self._layer_drift_delta: Dict[int, float] = {}

    def record_repair_attempt(self, layer_idx: int, step: int, drift_before: float, drift_after: float):
        """
        Called once per repair event.
        """
        delta = drift_before - drift_after  # positive = improvement
        with self._lock:
            self._activations += 1
            self._layer_activations[layer_idx] = self._layer_activations.get(layer_idx, 0) + 1
            self._layer_drift_delta[layer_idx] = (
                self._layer_drift_delta.get(layer_idx, 0.0) + delta
            )

            # Record persistence (how many steps since last repair)
            if layer_idx in self._last_repair_step:
                persistence = step - self._last_repair_step[layer_idx]
                if layer_idx not in self._layer_persistence:
                    self._layer_persistence[layer_idx] = []
                self._layer_persistence[layer_idx].append(persistence)
            
            self._last_repair_step[layer_idx] = step

            if delta >= self.EFFECTIVENESS_DELTA:
                self._effective += 1
                self._total_drift_reduction += delta
                self._layer_effective[layer_idx] = self._layer_effective.get(layer_idx, 0) + 1
                return True
            else:
                self._ineffective += 1
                return False

    def record_stabilization_step(self, layer_idx: int, drift: float):
        """Track drift after repair to calculate stabilization half-life."""
        with self._lock:
            if layer_idx not in self._drift_history:
                self._drift_history[layer_idx] = deque(maxlen=self.history_len)
            self._drift_history[layer_idx].append(drift)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._activations, 1)
            
            # Calculate mean persistence across all layers
            all_pers = []
            for p_list in self._layer_persistence.values():
                all_pers.extend(p_list)
            avg_persistence = sum(all_pers) / len(all_pers) if all_pers else 0.0

            return {
                "total_activations": self._activations,
                "effective_repairs": self._effective,
                "ineffective_repairs": self._ineffective,
                "effectiveness_rate": round(self._effective / total, 4),
                "avg_drift_reduction": round(
                    self._total_drift_reduction / max(self._effective, 1), 6
                ),
                "avg_recovery_persistence": round(avg_persistence, 2),
                "stabilization_quality": self._calculate_global_stabilization()
            }

    def _calculate_global_stabilization(self) -> float:
        """Heuristic: fraction of layers with stable/decreasing drift trends."""
        stable_count = 0
        if not self._drift_history: return 0.0
        for layer_idx, history in self._drift_history.items():
            if len(history) < 4: continue
            # If last half mean <= first half mean, it's stabilizing
            half = len(history) // 2
            first_half = list(history)[:half]
            last_half  = list(history)[half:]
            if (sum(last_half)/len(last_half)) <= (sum(first_half)/len(first_half)) * 1.05:
                stable_count += 1
        return round(stable_count / len(self._drift_history), 4)

    def get_layer_effectiveness(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            result = {}
            for layer_idx, acts in self._layer_activations.items():
                eff = self._layer_effective.get(layer_idx, 0)
                delta = self._layer_drift_delta.get(layer_idx, 0.0)
                pers = self._layer_persistence.get(layer_idx, [])
                result[layer_idx] = {
                    "activations": acts,
                    "effective": eff,
                    "rate": round(eff / max(acts, 1), 4),
                    "total_drift_reduction": round(delta, 6),
                    "avg_persistence": round(sum(pers)/len(pers), 2) if pers else 0.0
                }
            return result

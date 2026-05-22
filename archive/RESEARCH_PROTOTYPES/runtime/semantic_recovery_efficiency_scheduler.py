"""
STAGE 2 - SDR: Semantic Recovery Efficiency Scheduler
Phase 39.4 - Semantic Drift Reduction

Optimizes the deployment of semantic recovery windows to maximize
drift reduction while minimizing dense computation overhead.
"""
import threading
from typing import Dict, Any, List


class SemanticRecoveryEfficiencyScheduler:
    """
    Decides which layers 'deserve' recovery windows based on their
    contribution to global drift and their past recovery success rate.
    """
    MAX_CONCURRENT_RECOVERIES = 8
    MIN_DRIFT_TO_SCHEDULE     = 0.05

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        # layer -> success rate of past recoveries
        self._success_rates: Dict[int, float] = {i: 0.8 for i in range(num_layers)}
        # tracking current scheduling decisions
        self._active_schedules: List[int] = []

    def schedule_recoveries(self, drift_map: Dict[int, float]) -> List[int]:
        """
        Selects top-N layers that need recovery and are likely to benefit from it.
        """
        with self._lock:
            # Score = current_drift * historical_success_rate
            scores = []
            for i in range(self.num_layers):
                drift = drift_map.get(i, 0.0)
                if drift < self.MIN_DRIFT_TO_SCHEDULE:
                    continue
                score = drift * self._success_rates.get(i, 0.5)
                scores.append((i, score))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            selected = [idx for idx, _ in scores[:self.MAX_CONCURRENT_RECOVERIES]]
            self._active_schedules = selected
            return selected

    def record_outcome(self, layer_idx: int, success: bool):
        """Update historical success rates to improve future scheduling."""
        with self._lock:
            alpha = 0.2 # Smoothing factor
            current = self._success_rates.get(layer_idx, 0.5)
            new_val = 1.0 if success else 0.0
            self._success_rates[layer_idx] = (1 - alpha) * current + alpha * new_val

    def get_efficiency_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_success = sum(self._success_rates.values()) / self.num_layers
            return {
                "avg_recovery_success_probability": round(avg_success, 4),
                "scheduled_layer_count": len(self._active_schedules),
                "scheduling_precision": round(sum(1 for s in self._success_rates.values() if s > 0.7) / self.num_layers, 4)
            }

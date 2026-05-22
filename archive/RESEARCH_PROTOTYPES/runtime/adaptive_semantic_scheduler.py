"""
STAGE 2 - ASS: Adaptive Semantic Scheduler
Phase 39.5 - Adaptive Semantic Scheduling

Proactively rebalances sparse execution, hybrid execution, and dense recovery zones
BEFORE semantic collapse emerges based on pressure forecasts.
"""
import threading
from typing import Dict, Any, List

class AdaptiveSemanticScheduler:
    """
    Takes inputs from the pressure estimator and historical stability
    to preemptively adjust the inference execution mode for each layer.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        # 'dense', 'hybrid', 'sparse'
        self._current_schedule: Dict[int, str] = {i: "sparse" for i in range(num_layers)}
        self._dense_duration: Dict[int, int] = {i: 0 for i in range(num_layers)}

    def update_schedule(self, step: int, pressure_map: Dict[int, float], dense_critical_map: Dict[int, bool]):
        with self._lock:
            for layer_idx in range(self.num_layers):
                pressure = pressure_map.get(layer_idx, 0.0)
                is_critical = dense_critical_map.get(layer_idx, False)
                
                if is_critical:
                    # Hard override: if it's already critical, it must be dense
                    self._current_schedule[layer_idx] = "dense"
                    self._dense_duration[layer_idx] = 5 # Force dense for at least 5 steps
                elif pressure > 0.8:
                    # High pressure: preemptively densify
                    self._current_schedule[layer_idx] = "dense"
                    self._dense_duration[layer_idx] = 3
                elif pressure > 0.5:
                    # Moderate pressure: use hybrid routing to slow degradation
                    if self._current_schedule[layer_idx] == "sparse":
                        self._current_schedule[layer_idx] = "hybrid"
                else:
                    # Low pressure
                    if self._dense_duration[layer_idx] > 0:
                        self._dense_duration[layer_idx] -= 1
                    else:
                        # Smooth transition back to sparse
                        if self._current_schedule[layer_idx] == "dense":
                            self._current_schedule[layer_idx] = "hybrid"
                        else:
                            self._current_schedule[layer_idx] = "sparse"

    def get_schedule(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._current_schedule)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            modes = list(self._current_schedule.values())
            return {
                "sparse_layers": modes.count("sparse"),
                "hybrid_layers": modes.count("hybrid"),
                "dense_layers": modes.count("dense"),
                "proactive_dense_ratio": round(modes.count("dense") / max(self.num_layers, 1), 4)
            }

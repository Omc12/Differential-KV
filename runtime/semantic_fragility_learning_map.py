"""
STAGE 2 - ASI: Semantic Fragility Learning Map
Phase 39.6 - Adaptive Semantic Intelligence

Learns which transformer regions become unstable most often.
Evolves dynamically over time.
"""
import threading
from typing import Dict, Any, List

class SemanticFragilityLearningMap:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        # Layer -> Fragility Score (0.0 to 1.0)
        self._layer_fragility: Dict[int, float] = {i: 0.1 for i in range(num_layers)}
        self._layer_collapse_counts: Dict[int, int] = {i: 0 for i in range(num_layers)}

    def record_collapse(self, layer_idx: int):
        with self._lock:
            self._layer_collapse_counts[layer_idx] += 1
            # Increase fragility sharply on collapse
            self._layer_fragility[layer_idx] = min(1.0, self._layer_fragility[layer_idx] + 0.2)

    def record_stable_step(self, layer_idx: int):
        with self._lock:
            # Decay fragility slowly during stable periods
            self._layer_fragility[layer_idx] = max(0.01, self._layer_fragility[layer_idx] * 0.999)

    def is_fragile(self, layer_idx: int, threshold: float = 0.6) -> bool:
        with self._lock:
            return self._layer_fragility[layer_idx] > threshold

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            fragile_count = sum(1 for f in self._layer_fragility.values() if f > 0.6)
            avg_fragility = sum(self._layer_fragility.values()) / max(self.num_layers, 1)
            return {
                "fragile_layers": fragile_count,
                "avg_fragility_score": round(avg_fragility, 4),
                "total_collapses_tracked": sum(self._layer_collapse_counts.values())
            }

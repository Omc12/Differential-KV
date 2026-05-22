"""
STAGE 2 - SDR: Anchor Reinforcement Engine
Phase 39.4 - Semantic Drift Reduction

Strengthens semantic anchors by tracking degradation and forcing
periodic anchor-guided stabilization.
"""
import threading
from typing import Dict, Any, List


class AnchorReinforcementEngine:
    """
    Monitors the 'freshness' and reliability of semantic anchors.
    If an anchor has been used for too many sparse steps without reinforcement,
    it triggers a targeted dense update.
    """
    DEGRADATION_THRESHOLD = 20  # steps before anchor is considered 'stale'

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        # layer -> steps since last reinforcement
        self._anchor_age: Dict[int, int] = {i: 0 for i in range(num_layers)}
        # layer -> total reinforcements
        self._reinforcement_count: Dict[int, int] = {i: 0 for i in range(num_layers)}
        # tracking effectiveness: layer -> drift reduction sum
        self._reinforcement_impact: Dict[int, float] = {i: 0.0 for i in range(num_layers)}

    def record_step(self, layer_idx: int, is_dense_step: bool):
        """Update anchor age based on execution mode."""
        with self._lock:
            if is_dense_step:
                # Dense execution reinforces anchors automatically
                self._anchor_age[layer_idx] = 0
                self._reinforcement_count[layer_idx] += 1
            else:
                self._anchor_age[layer_idx] += 1

    def needs_reinforcement(self, layer_idx: int) -> bool:
        """Determines if the anchor for this layer is too stale."""
        with self._lock:
            return self._anchor_age.get(layer_idx, 0) >= self.DEGRADATION_THRESHOLD

    def record_reinforcement_impact(self, layer_idx: int, drift_reduction: float):
        """Measures how much a targeted reinforcement actually helped."""
        with self._lock:
            self._reinforcement_impact[layer_idx] += drift_reduction

    def get_anchor_health(self) -> Dict[str, Any]:
        with self._lock:
            stale_count = sum(1 for a in self._anchor_age.values() if a >= self.DEGRADATION_THRESHOLD)
            avg_age = sum(self._anchor_age.values()) / self.num_layers
            return {
                "stale_anchors": stale_count,
                "avg_anchor_age": round(avg_age, 2),
                "total_reinforcements": sum(self._reinforcement_count.values()),
                "total_impact": round(sum(self._reinforcement_impact.values()), 4)
            }

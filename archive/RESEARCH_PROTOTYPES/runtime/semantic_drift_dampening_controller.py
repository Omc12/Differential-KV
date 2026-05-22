"""
STAGE 2 - SDR: Semantic Drift Dampening Controller
Phase 39.4 - Semantic Drift Reduction

Prevents semantic oscillation loops (collapse -> repair -> collapse)
by implementing adaptive cooldowns and progressive recovery escalation.
"""
import threading
import time
from typing import Dict, Any, List


class SemanticDriftDampeningController:
    """
    Manages the intensity and frequency of semantic repairs to ensure stability.
    If a layer collapses too quickly after repair, it escalates to longer
    dense-stabilization windows.
    """
    MIN_STABILIZATION_WINDOW = 4   # steps
    MAX_STABILIZATION_WINDOW = 32  # steps
    OSCILLATION_THRESHOLD    = 10  # steps between repairs considered 'oscillation'

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        # layer -> current stabilization window size
        self._stabilization_windows: Dict[int, int] = {i: self.MIN_STABILIZATION_WINDOW for i in range(num_layers)}
        # layer -> remaining steps in cooldown
        self._cooldown_remaining: Dict[int, int] = {i: 0 for i in range(num_layers)}
        # layer -> timestamp of last repair
        self._last_repair_step: Dict[int, int] = {i: 0 for i in range(num_layers)}
        # layer -> oscillation count
        self._oscillation_count: Dict[int, int] = {i: 0 for i in range(num_layers)}

    def should_allow_repair(self, layer_idx: int, current_step: int) -> bool:
        """Determines if a repair is permitted or if dampening is active."""
        with self._lock:
            # If in mandatory stabilization cooldown, block repair
            if self._cooldown_remaining.get(layer_idx, 0) > 0:
                return False
            return True

    def record_repair_event(self, layer_idx: int, current_step: int):
        """Called when a repair is actually executed."""
        with self._lock:
            last_step = self._last_repair_step.get(layer_idx, 0)
            interval = current_step - last_step
            
            if interval < self.OSCILLATION_THRESHOLD and last_step > 0:
                # Oscillation detected! Escalate dampening.
                self._oscillation_count[layer_idx] += 1
                new_window = min(
                    self._stabilization_windows[layer_idx] * 2,
                    self.MAX_STABILIZATION_WINDOW
                )
                self._stabilization_windows[layer_idx] = new_window
            else:
                # Stable period. Slowly decay dampening.
                self._oscillation_count[layer_idx] = max(0, self._oscillation_count[layer_idx] - 1)
                if self._oscillation_count[layer_idx] == 0:
                    self._stabilization_windows[layer_idx] = max(
                        self._stabilization_windows[layer_idx] // 2,
                        self.MIN_STABILIZATION_WINDOW
                    )

            self._last_repair_step[layer_idx] = current_step
            # Trigger mandatory dense-cooldown (dampening)
            self._cooldown_remaining[layer_idx] = self._stabilization_windows[layer_idx]

    def update_step(self):
        """Tick down cooldowns at each step."""
        with self._lock:
            for i in range(self.num_layers):
                if self._cooldown_remaining[i] > 0:
                    self._cooldown_remaining[i] -= 1

    def get_dampening_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_cooldowns = sum(1 for c in self._cooldown_remaining.values() if c > 0)
            avg_window = sum(self._stabilization_windows.values()) / self.num_layers
            return {
                "active_cooldown_layers": total_cooldowns,
                "avg_stabilization_window": round(avg_window, 2),
                "total_oscillations": sum(self._oscillation_count.values())
            }

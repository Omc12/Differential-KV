"""
STAGE 2 - HSZ: Semantic Recovery Zone Controller
Phase 39.3 - Hybrid Semantic Zoning

Manages targeted dense recovery windows.
Dense execution is semantically justified and targeted —
NOT a blanket fallback.
"""
import time
import threading
from typing import Dict, Any, List


class SemanticRecoveryZoneController:
    """
    Controls when and where dense recovery windows are opened.
    Tracks per-layer recovery effectiveness over time.
    """
    # Consecutive steps of high drift before opening a dense recovery window
    DRIFT_STREAK_TRIGGER = 3
    # Number of dense steps in a recovery window
    RECOVERY_WINDOW_LEN  = 4

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.Lock()
        # Per-layer consecutive high-drift streak
        self._drift_streak: Dict[int, int] = {}
        # Per-layer active recovery window remaining steps
        self._recovery_rem: Dict[int, int] = {}
        self._total_windows_opened = 0
        self._windows_by_layer: Dict[int, int] = {}

    def update(self, layer_idx: int, drift: float, drift_threshold: float) -> bool:
        """
        Update drift tracking for a layer. Returns True if a dense recovery
        window should be active this step.
        """
        with self._lock:
            streak = self._drift_streak.get(layer_idx, 0)
            rem = self._recovery_rem.get(layer_idx, 0)

            if drift >= drift_threshold:
                streak += 1
            else:
                streak = max(0, streak - 1)

            if rem > 0:
                # Already in a recovery window
                rem -= 1
                self._recovery_rem[layer_idx] = rem
                self._drift_streak[layer_idx] = streak
                return True

            if streak >= self.DRIFT_STREAK_TRIGGER:
                # Open a new recovery window
                self._recovery_rem[layer_idx] = self.RECOVERY_WINDOW_LEN
                self._drift_streak[layer_idx] = 0
                self._total_windows_opened += 1
                self._windows_by_layer[layer_idx] = (
                    self._windows_by_layer.get(layer_idx, 0) + 1
                )
                return True

            self._drift_streak[layer_idx] = streak
            return False

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_recovery_windows": self._total_windows_opened,
                "windows_by_layer": dict(self._windows_by_layer),
            }

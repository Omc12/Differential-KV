"""
analysis/threshold_tracker.py — Task 3: Dynamic Threshold Systems

Tracks and evolves thresholds over time during sequence traversal.

Implements:
  1. RollingWindowTracker — tracks threshold over sliding window
  2. PercentileTracker    — sets threshold at Nth percentile of observations
  3. VarianceNormalizedTracker — normalizes observations by local variance
  4. AdaptivePercentileTracker — percentile clips and evolves percentile level
  5. PerLayerScaler        — per-layer adaptive scaling

Each tracker exposes:
  - update(value)        — add new observation
  - get_threshold()      — current computed threshold
  - get_history()        — full history of thresholds
  - summary()            — statistics dict
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class RollingWindowTracker:
    """
    Threshold = rolling_mean + k * rolling_std over last N observations.

    Parameters
    ----------
    window_size : int   — number of recent observations
    k           : float — multiplier on standard deviation
    floor       : float — minimum threshold (prevents zero threshold)
    """

    def __init__(self, window_size: int = 64, k: float = 2.0, floor: float = 0.01):
        self.window_size = window_size
        self.k           = k
        self.floor       = floor
        self._window: Deque[float] = deque(maxlen=window_size)
        self._threshold_history: List[float] = []

    def update(self, value: float):
        self._window.append(value)
        self._threshold_history.append(self.get_threshold())

    def get_threshold(self) -> float:
        if len(self._window) < 2:
            return self.floor
        arr  = np.array(self._window)
        mean = float(arr.mean())
        std  = float(arr.std())
        return max(self.floor, mean + self.k * std)

    def get_history(self) -> List[float]:
        return list(self._threshold_history)

    def summary(self) -> Dict:
        if not self._threshold_history:
            return {"tracker": "RollingWindow", "num_observations": 0}
        arr = np.array(self._threshold_history)
        return {
            "tracker":              "RollingWindow",
            "window_size":          self.window_size,
            "k":                    self.k,
            "num_observations":     len(self._threshold_history),
            "mean_threshold":       round(float(arr.mean()), 5),
            "std_threshold":        round(float(arr.std()),  5),
            "min_threshold":        round(float(arr.min()),  5),
            "max_threshold":        round(float(arr.max()),  5),
            "final_threshold":      round(self.get_threshold(), 5),
        }


class PercentileTracker:
    """
    Threshold = Nth percentile of all historical observations.

    Builds a running buffer of observations and recomputes the Nth
    percentile as the threshold. Simple but effective.

    Parameters
    ----------
    percentile  : float — e.g. 90 means only top 10% trigger
    buffer_size : int   — max history size (older values dropped)
    """

    def __init__(self, percentile: float = 90.0, buffer_size: int = 1000):
        self.percentile  = percentile
        self.buffer_size = buffer_size
        self._buffer: Deque[float] = deque(maxlen=buffer_size)
        self._threshold_history: List[float] = []

    def update(self, value: float):
        self._buffer.append(value)
        self._threshold_history.append(self.get_threshold())

    def get_threshold(self) -> float:
        if not self._buffer:
            return 0.0
        return float(np.percentile(list(self._buffer), self.percentile))

    def get_history(self) -> List[float]:
        return list(self._threshold_history)

    def summary(self) -> Dict:
        if not self._threshold_history:
            return {"tracker": "Percentile", "num_observations": 0}
        arr = np.array(self._threshold_history)
        return {
            "tracker":         "Percentile",
            "percentile":      self.percentile,
            "num_observations": len(self._threshold_history),
            "final_threshold": round(self.get_threshold(), 5),
            "mean_threshold":  round(float(arr.mean()), 5),
        }


class VarianceNormalizedTracker:
    """
    Normalizes each observation by local variance before thresholding.
    Prevents false triggers during globally high-variance periods.

    normalized_value = (value - local_mean) / (local_std + eps)
    threshold        = fixed_z_score (e.g. 2.0 sigma)
    """

    def __init__(self, z_score: float = 2.0, window_size: int = 64, eps: float = 1e-6):
        self.z_score     = z_score
        self.window_size = window_size
        self.eps         = eps
        self._window: Deque[float] = deque(maxlen=window_size)
        self._normalized_history: List[float] = []

    def update(self, value: float) -> float:
        """Returns normalized value for this observation."""
        self._window.append(value)
        return self.normalize(value)

    def normalize(self, value: float) -> float:
        if len(self._window) < 2:
            return 0.0
        arr  = np.array(self._window)
        mean = float(arr.mean())
        std  = float(arr.std())
        return (value - mean) / (std + self.eps)

    def get_threshold(self) -> float:
        return self.z_score

    def exceeds_threshold(self, value: float) -> bool:
        return self.normalize(value) > self.z_score

    def get_history(self) -> List[float]:
        return list(self._normalized_history)

    def summary(self) -> Dict:
        if len(self._window) < 2:
            return {"tracker": "VarianceNormalized", "num_observations": len(self._window)}
        arr  = np.array(self._window)
        return {
            "tracker":         "VarianceNormalized",
            "z_score":         self.z_score,
            "window_size":     self.window_size,
            "num_observations": len(self._window),
            "current_mean":    round(float(arr.mean()), 5),
            "current_std":     round(float(arr.std()),  5),
            "threshold_z":     self.z_score,
        }


class AdaptivePercentileTracker:
    """
    Evolves the percentile level itself over time.
    If trigger rate is too high: raise percentile (fewer triggers).
    If trigger rate is too low: lower percentile (more triggers).
    Targets a specific trigger rate.

    Parameters
    ----------
    target_rate   : float — desired trigger fraction (e.g. 0.05 = 5%)
    initial_pct   : float — starting percentile
    adjust_rate   : float — how fast to adjust (per observation)
    buffer_size   : int
    """

    def __init__(self, target_rate: float = 0.05, initial_pct: float = 90.0,
                 adjust_rate: float = 0.1, buffer_size: int = 200):
        self.target_rate   = target_rate
        self.percentile    = initial_pct
        self.adjust_rate   = adjust_rate
        self._buffer: Deque[float] = deque(maxlen=buffer_size)
        self._trigger_count = 0
        self._total_count   = 0
        self._pct_history: List[float] = []

    def update(self, value: float) -> bool:
        """Update tracker with new value. Returns True if value triggers."""
        self._buffer.append(value)
        self._total_count += 1
        threshold = self.get_threshold()
        triggered = value > threshold

        if triggered:
            self._trigger_count += 1

        # Adapt percentile toward target rate
        if self._total_count % 20 == 0 and self._total_count > 0:
            current_rate = self._trigger_count / self._total_count
            if current_rate > self.target_rate:
                self.percentile = min(99.9, self.percentile + self.adjust_rate)
            elif current_rate < self.target_rate:
                self.percentile = max(50.0, self.percentile - self.adjust_rate)
            self._trigger_count = 0
            self._total_count   = 0

        self._pct_history.append(self.percentile)
        return triggered

    def get_threshold(self) -> float:
        if len(self._buffer) < 10:
            return 0.0
        return float(np.percentile(list(self._buffer), self.percentile))

    def summary(self) -> Dict:
        if not self._pct_history:
            return {"tracker": "AdaptivePercentile"}
        arr = np.array(self._pct_history)
        return {
            "tracker":            "AdaptivePercentile",
            "target_rate":        self.target_rate,
            "current_percentile": round(self.percentile, 2),
            "mean_percentile":    round(float(arr.mean()), 2),
            "final_threshold":    round(self.get_threshold(), 5),
        }


class PerLayerScaler:
    """
    Maintains a separate threshold tracker per layer.
    Allows each layer to have independently evolving thresholds.

    Parameters
    ----------
    tracker_type : str — 'rolling', 'percentile', 'variance', 'adaptive'
    tracker_kwargs : dict — kwargs forwarded to tracker constructor
    """

    def __init__(self, tracker_type: str = "rolling", **tracker_kwargs):
        self.tracker_type   = tracker_type
        self.tracker_kwargs = tracker_kwargs
        self._trackers: Dict[int, object] = {}

    def _get_tracker(self, layer_idx: int):
        if layer_idx not in self._trackers:
            if self.tracker_type == "rolling":
                self._trackers[layer_idx] = RollingWindowTracker(**self.tracker_kwargs)
            elif self.tracker_type == "percentile":
                self._trackers[layer_idx] = PercentileTracker(**self.tracker_kwargs)
            elif self.tracker_type == "variance":
                self._trackers[layer_idx] = VarianceNormalizedTracker(**self.tracker_kwargs)
            elif self.tracker_type == "adaptive":
                self._trackers[layer_idx] = AdaptivePercentileTracker(**self.tracker_kwargs)
            else:
                raise ValueError(f"Unknown tracker type: {self.tracker_type}")
        return self._trackers[layer_idx]

    def update(self, layer_idx: int, value: float):
        self._get_tracker(layer_idx).update(value)

    def get_threshold(self, layer_idx: int) -> float:
        return self._get_tracker(layer_idx).get_threshold()

    def summary_all(self) -> Dict[int, Dict]:
        return {li: t.summary() for li, t in self._trackers.items()}

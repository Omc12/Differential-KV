"""
anchor_logic/adaptive_policies.py

Phase 2: Smarter Adaptive Anchor Trigger Policies.

Root cause of Phase 1 failure:
  Raw L2 norm of [2, heads, head_dim] tensor grows with tensor size.
  A [2, 32, 128] unit-normal tensor has L2 norm ~90.
  Any fixed threshold < 90 triggers on every token.
  Fix: normalize by sqrt(num_elements) → RMS-equivalent threshold.

Five trigger policies (all normalize correctly):

1. AbsoluteNormalized   - threshold on RMS of delta (not raw L2)
2. RelativeChange       - threshold on ||delta|| / ||anchor||
3. RollingVariance      - trigger if delta_rms > rolling_mean + k * rolling_std
4. EMABased             - trigger if delta_rms > EMA * sensitivity_factor
5. LayerNormalized      - normalize delta by anchor statistics before comparing

All policies expose:
  - should_anchor(token_idx, kv, last_anchor_kv, last_anchor_idx)
  - get_stats()  → dict of internal state for observability
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import torch

from .strategies import AnchorDecision


# ── Base class ──────────────────────────────────────────────────────────────

class AdaptivePolicyBase:
    """Base class for all adaptive anchor trigger policies."""

    def should_anchor(self, token_idx: int, kv: torch.Tensor,
                      last_anchor_kv: torch.Tensor,
                      last_anchor_idx: int) -> AnchorDecision:
        raise NotImplementedError

    def get_stats(self) -> Dict:
        return {}

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        """Root-mean-square: L2 norm / sqrt(num_elements). Scale-invariant."""
        return (x.float().norm() / math.sqrt(x.numel())).item()

    @staticmethod
    def _relative_change(delta: torch.Tensor, anchor: torch.Tensor) -> float:
        """||delta||_2 / (||anchor||_2 + eps)"""
        return (delta.float().norm() / (anchor.float().norm() + 1e-9)).item()


# ── Policy 1: AbsoluteNormalized ─────────────────────────────────────────────

class AbsoluteNormalizedPolicy(AdaptivePolicyBase):
    """
    Trigger if RMS(delta) > threshold.

    This is the corrected version of Phase 1's absolute threshold.
    RMS normalizes by tensor size, so the threshold is scale-invariant.

    Typical KV RMS for unit-normal FP16: ~1.0
    Threshold of 0.1-0.5 gives reasonable anchor density.

    Parameters
    ----------
    threshold    : float  — RMS trigger level
    max_interval : int    — hard periodic fallback
    min_interval : int    — minimum tokens between anchors
    """

    def __init__(self, threshold: float = 0.3, max_interval: int = 128,
                 min_interval: int = 8):
        self.threshold    = threshold
        self.max_interval = max_interval
        self.min_interval = min_interval
        self._trigger_counts = {"periodic": 0, "magnitude_rms": 0, "none": 0}
        self._rms_history: List[float] = []

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        delta = kv.float() - last_anchor_kv.float()
        rms   = self._rms(delta)
        self._rms_history.append(rms)

        if rms > self.threshold:
            self._trigger_counts["magnitude_rms"] += 1
            return AnchorDecision(is_anchor=True, reason="magnitude_rms",
                                  delta_norm=rms)
        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

    def get_stats(self):
        return {
            "policy": "AbsoluteNormalized",
            "threshold": self.threshold,
            "trigger_counts": dict(self._trigger_counts),
            "mean_rms": (sum(self._rms_history) / len(self._rms_history)
                         if self._rms_history else 0.0),
        }


# ── Policy 2: RelativeChange ─────────────────────────────────────────────────

class RelativeChangePolicy(AdaptivePolicyBase):
    """
    Trigger if ||delta||_2 / ||anchor||_2 > threshold.

    Naturally adapts to anchor magnitude.
    A threshold of 0.1 = trigger when delta is 10% of anchor magnitude.
    Much more robust across different KV scales than absolute thresholds.
    """

    def __init__(self, threshold: float = 0.15, max_interval: int = 128,
                 min_interval: int = 8):
        self.threshold    = threshold
        self.max_interval = max_interval
        self.min_interval = min_interval
        self._trigger_counts = {"periodic": 0, "relative_change": 0, "none": 0}
        self._rel_history: List[float] = []

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        delta = kv.float() - last_anchor_kv.float()
        rel   = self._relative_change(delta, last_anchor_kv)
        self._rel_history.append(rel)

        if rel > self.threshold:
            self._trigger_counts["relative_change"] += 1
            return AnchorDecision(is_anchor=True, reason="relative_change",
                                  delta_norm=rel,
                                  reconstruction_error_estimate=rel)
        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none", delta_norm=rel)

    def get_stats(self):
        return {
            "policy": "RelativeChange",
            "threshold": self.threshold,
            "trigger_counts": dict(self._trigger_counts),
            "mean_relative": (sum(self._rel_history) / len(self._rel_history)
                              if self._rel_history else 0.0),
        }


# ── Policy 3: RollingVariance ─────────────────────────────────────────────────

class RollingVariancePolicy(AdaptivePolicyBase):
    """
    Trigger if delta_rms > rolling_mean + k * rolling_std.

    Adapts to the local distribution of deltas.
    Only triggers on genuine statistical outliers.
    Prevents anchor explosion in stable regions.
    Triggers appropriately at sharp distribution shifts.

    Parameters
    ----------
    k            : float — number of standard deviations above mean
    window_size  : int   — rolling window length (in tokens)
    warmup       : int   — tokens to observe before triggering adaptively
    max_interval : int   — hard fallback
    min_interval : int
    """

    def __init__(self, k: float = 2.0, window_size: int = 64,
                 warmup: int = 32, max_interval: int = 256, min_interval: int = 8):
        self.k            = k
        self.window_size  = window_size
        self.warmup       = warmup
        self.max_interval = max_interval
        self.min_interval = min_interval

        self._window: Deque[float] = deque(maxlen=window_size)
        self._trigger_counts = {"periodic": 0, "rolling_outlier": 0,
                                "warmup_periodic": 0, "none": 0}
        self._threshold_history: List[float] = []

    def _rolling_stats(self):
        if len(self._window) < 2:
            return 0.0, 1.0
        mean = sum(self._window) / len(self._window)
        var  = sum((x - mean) ** 2 for x in self._window) / len(self._window)
        return mean, math.sqrt(var)

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        delta = kv.float() - last_anchor_kv.float()
        rms   = self._rms(delta)

        # During warmup: periodic triggers at half max_interval
        if len(self._window) < self.warmup:
            self._window.append(rms)
            if tokens_since >= self.max_interval // 2:
                self._trigger_counts["warmup_periodic"] += 1
                return AnchorDecision(is_anchor=True, reason="warmup_periodic",
                                      delta_norm=rms)
            return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

        mean, std = self._rolling_stats()
        dynamic_threshold = mean + self.k * std
        self._threshold_history.append(dynamic_threshold)
        self._window.append(rms)

        if rms > dynamic_threshold:
            self._trigger_counts["rolling_outlier"] += 1
            return AnchorDecision(is_anchor=True, reason="rolling_outlier",
                                  delta_norm=rms,
                                  reconstruction_error_estimate=rms / (mean + 1e-9))
        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

    def get_stats(self):
        mean, std = self._rolling_stats()
        return {
            "policy": "RollingVariance",
            "k": self.k,
            "window_size": self.window_size,
            "trigger_counts": dict(self._trigger_counts),
            "current_rolling_mean": round(mean, 5),
            "current_rolling_std": round(std, 5),
            "mean_threshold": (sum(self._threshold_history) / len(self._threshold_history)
                               if self._threshold_history else 0.0),
        }


# ── Policy 4: EMABased ───────────────────────────────────────────────────────

class EMAPolicy(AdaptivePolicyBase):
    """
    Maintain an Exponential Moving Average of delta RMS.
    Trigger when current delta_rms > EMA * sensitivity_factor.

    EMA smoothly tracks the "expected" delta level.
    A sudden spike above sensitivity * EMA triggers an anchor.
    Recovers quickly after the spike — no anchor clustering.

    Parameters
    ----------
    alpha              : float — EMA decay (0.1=slow adaptation, 0.9=fast)
    sensitivity_factor : float — trigger when rms > ema * factor
    max_interval       : int
    min_interval       : int
    """

    def __init__(self, alpha: float = 0.1, sensitivity_factor: float = 2.5,
                 max_interval: int = 256, min_interval: int = 8):
        self.alpha              = alpha
        self.sensitivity_factor = sensitivity_factor
        self.max_interval       = max_interval
        self.min_interval       = min_interval

        self._ema: Optional[float] = None
        self._trigger_counts = {"periodic": 0, "ema_spike": 0, "none": 0}
        self._ema_history: List[float] = []

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        delta = kv.float() - last_anchor_kv.float()
        rms   = self._rms(delta)

        # Initialize EMA on first observation
        if self._ema is None:
            self._ema = rms
            return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

        trigger_level = self._ema * self.sensitivity_factor
        self._ema_history.append(self._ema)

        if rms > trigger_level:
            # Update EMA AFTER decision (spike shouldn't corrupt baseline)
            self._ema = self.alpha * rms + (1 - self.alpha) * self._ema
            self._trigger_counts["ema_spike"] += 1
            return AnchorDecision(is_anchor=True, reason="ema_spike",
                                  delta_norm=rms,
                                  reconstruction_error_estimate=rms / (self._ema + 1e-9))
        # Normal update
        self._ema = self.alpha * rms + (1 - self.alpha) * self._ema
        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

    def get_stats(self):
        return {
            "policy": "EMA",
            "alpha": self.alpha,
            "sensitivity_factor": self.sensitivity_factor,
            "trigger_counts": dict(self._trigger_counts),
            "current_ema": round(self._ema, 5) if self._ema else 0.0,
        }


# ── Policy 5: LayerNormalized ─────────────────────────────────────────────────

class LayerNormalizedPolicy(AdaptivePolicyBase):
    """
    Normalize the delta by the anchor's per-dimension statistics
    before comparing to threshold. Inspired by LayerNorm.

    delta_normalized = (kv - anchor) / (std(anchor) + eps)
    trigger if mean(|delta_normalized|) > threshold

    Properties:
    - Invariant to anchor scale
    - Invariant to anchor magnitude
    - Sensitive to changes relative to the anchor's own distribution

    Parameters
    ----------
    threshold    : float — mean normalized deviation trigger
    max_interval : int
    min_interval : int
    """

    def __init__(self, threshold: float = 0.5, max_interval: int = 128,
                 min_interval: int = 8):
        self.threshold    = threshold
        self.max_interval = max_interval
        self.min_interval = min_interval
        self._trigger_counts = {"periodic": 0, "layernorm_deviation": 0, "none": 0}
        self._deviation_history: List[float] = []

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        anchor_f = last_anchor_kv.float()
        kv_f     = kv.float()
        delta_f  = kv_f - anchor_f

        # Normalize by anchor std (per-tensor, like LayerNorm)
        anchor_std = anchor_f.std().item()
        if anchor_std < 1e-9:
            normalized_dev = delta_f.abs().mean().item()
        else:
            normalized_dev = (delta_f / (anchor_std + 1e-9)).abs().mean().item()

        self._deviation_history.append(normalized_dev)

        if normalized_dev > self.threshold:
            self._trigger_counts["layernorm_deviation"] += 1
            return AnchorDecision(is_anchor=True, reason="layernorm_deviation",
                                  delta_norm=normalized_dev,
                                  reconstruction_error_estimate=normalized_dev)
        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none",
                              delta_norm=normalized_dev)

    def get_stats(self):
        return {
            "policy": "LayerNormalized",
            "threshold": self.threshold,
            "trigger_counts": dict(self._trigger_counts),
            "mean_deviation": (sum(self._deviation_history) / len(self._deviation_history)
                               if self._deviation_history else 0.0),
        }


# ── Policy 6: DynamicThreshold ────────────────────────────────────────────────

class DynamicThresholdPolicy(AdaptivePolicyBase):
    """
    Task 3: Thresholds that evolve over time during sequence traversal.

    Uses a tracker from analysis.threshold_tracker to maintain and
    evolve the threshold token-by-token.

    Parameters
    ----------
    tracker_type : str — 'rolling', 'percentile', 'variance', 'adaptive'
    max_interval : int
    min_interval : int
    kwargs       : passed to the tracker constructor
    """

    def __init__(self, tracker_type: str = "adaptive", max_interval: int = 256,
                 min_interval: int = 8, **kwargs):
        from analysis.threshold_tracker import AdaptivePercentileTracker, RollingWindowTracker, PercentileTracker, VarianceNormalizedTracker

        self.tracker_type = tracker_type
        if tracker_type == "adaptive":
            self.tracker = AdaptivePercentileTracker(**kwargs)
        elif tracker_type == "rolling":
            self.tracker = RollingWindowTracker(**kwargs)
        elif tracker_type == "percentile":
            self.tracker = PercentileTracker(**kwargs)
        elif tracker_type == "variance":
            self.tracker = VarianceNormalizedTracker(**kwargs)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

        self.max_interval = max_interval
        self.min_interval = min_interval
        self._trigger_counts = {"periodic": 0, "dynamic_trigger": 0, "none": 0}

    def should_anchor(self, token_idx, kv, last_anchor_kv, last_anchor_idx):
        tokens_since = token_idx - last_anchor_idx
        if tokens_since < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")
        if tokens_since >= self.max_interval:
            self._trigger_counts["periodic"] += 1
            return AnchorDecision(is_anchor=True, reason="periodic")

        delta = kv.float() - last_anchor_kv.float()
        rms   = self._rms(delta)

        # Update tracker and check for trigger
        if self.tracker_type == "adaptive":
            triggered = self.tracker.update(rms)
        elif self.tracker_type == "variance":
            triggered = self.tracker.exceeds_threshold(rms)
            self.tracker.update(rms)
        else:
            triggered = rms > self.tracker.get_threshold()
            self.tracker.update(rms)

        if triggered:
            self._trigger_counts["dynamic_trigger"] += 1
            return AnchorDecision(is_anchor=True, reason="dynamic_trigger",
                                  delta_norm=rms,
                                  reconstruction_error_estimate=rms / (self.tracker.get_threshold() + 1e-9))

        self._trigger_counts["none"] += 1
        return AnchorDecision(is_anchor=False, reason="none", delta_norm=rms)

    def get_stats(self):
        stats = {
            "policy": f"DynamicThreshold({self.tracker_type})",
            "trigger_counts": dict(self._trigger_counts),
        }
        stats.update(self.tracker.summary())
        return stats


# ── Factory ──────────────────────────────────────────────────────────────────

def make_policy(name: str, **kwargs) -> AdaptivePolicyBase:
    """
    Factory for all adaptive policies.

    Names: 'absolute_norm', 'relative_change', 'rolling_variance',
           'ema', 'layer_normalized', 'dynamic'
    """
    registry = {
        "absolute_norm":    AbsoluteNormalizedPolicy,
        "relative_change":  RelativeChangePolicy,
        "rolling_variance": RollingVariancePolicy,
        "ema":              EMAPolicy,
        "layer_normalized": LayerNormalizedPolicy,
        "dynamic":          DynamicThresholdPolicy,
    }
    if name not in registry:
        raise ValueError(f"Unknown policy '{name}'. Options: {list(registry)}")
    return registry[name](**kwargs)


# ── Convenience presets ──────────────────────────────────────────────────────

POLICY_PRESETS = {
    "conservative": ("rolling_variance", {"k": 3.0, "window_size": 64,
                                          "max_interval": 512, "min_interval": 16}),
    "balanced":     ("ema",              {"alpha": 0.1, "sensitivity_factor": 2.5,
                                          "max_interval": 256, "min_interval": 8}),
    "aggressive":   ("relative_change",  {"threshold": 0.05, "max_interval": 64,
                                          "min_interval": 4}),
    "layernorm":    ("layer_normalized", {"threshold": 0.4, "max_interval": 256,
                                          "min_interval": 8}),
    "adaptive_5pct": ("dynamic",          {"tracker_type": "adaptive", "target_rate": 0.05,
                                           "max_interval": 512, "min_interval": 8}),
}

"""
anchor_logic/layer_selector.py — Phase 2.5 Objective 2

Implements heterogeneous per-layer compression strategies.

Three modes:
  A. EarlyOnly       — compress layers 0..k, leave the rest at full-anchor
  B. Progressive     — aggressive early / balanced mid / conservative late
  C. PerLayerInterval — custom anchor interval per layer
  D. PerLayerAdaptive — each layer gets its own normalized adaptive policy
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import (
    EMAPolicy, RollingVariancePolicy, AbsoluteNormalizedPolicy,
    LayerNormalizedPolicy, DynamicThresholdPolicy,
)


# ── Strategy presets for named tiers ─────────────────────────────────────────

def _make_strategy(name: str):
    if name == "aggressive":
        return PeriodicAnchorStrategy(interval=128)
    elif name == "balanced":
        return EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                         max_interval=256, min_interval=8)
    elif name == "conservative":
        return RollingVariancePolicy(k=3.0, window_size=64,
                                      max_interval=512, min_interval=16)
    elif name == "full":
        # Dense periodic: anchor every 16 tokens
        return PeriodicAnchorStrategy(interval=16)
    elif name == "layernorm":
        return LayerNormalizedPolicy(threshold=0.4, max_interval=256, min_interval=8)
    elif name.startswith("periodic_"):
        interval = int(name.split("_")[1])
        return PeriodicAnchorStrategy(interval=interval)
    else:
        raise ValueError(f"Unknown strategy name: {name}")


def _make_adaptive(tracker_type: str = "adaptive", target_rate: float = 0.05):
    return DynamicThresholdPolicy(
        tracker_type=tracker_type, target_rate=target_rate,
        max_interval=512, min_interval=8
    )


# ── Selector classes ───────────────────────────────────────────────────────────

class EarlyOnlySelector:
    """
    Compress only the first `compress_layers` layers with `strategy`.
    All other layers use dense periodic anchoring (interval=16).
    """

    def __init__(self, num_layers: int, compress_layers: int,
                 strategy: str = "balanced"):
        self.num_layers      = num_layers
        self.compress_layers = compress_layers
        self._strategies     = {}
        for layer in range(num_layers):
            if layer < compress_layers:
                self._strategies[layer] = _make_strategy(strategy)
            else:
                self._strategies[layer] = PeriodicAnchorStrategy(interval=16)

    def get_strategy(self, layer_idx: int):
        return self._strategies.get(layer_idx, PeriodicAnchorStrategy(interval=16))

    def describe(self) -> Dict[int, str]:
        return {
            li: f"compressed({type(s).__name__})" if li < self.compress_layers else "dense"
            for li, s in self._strategies.items()
        }


class ProgressiveSelector:
    """
    Early layers → aggressive
    Middle layers → balanced
    Late layers → conservative
    """

    def __init__(self, num_layers: int,
                 early_frac: float = 0.25,
                 late_frac: float = 0.25):
        self.num_layers = num_layers
        early_cut = int(num_layers * early_frac)
        late_cut  = int(num_layers * (1.0 - late_frac))
        self._strategies = {}
        self._tiers = {}
        for layer in range(num_layers):
            if layer < early_cut:
                self._strategies[layer] = _make_strategy("aggressive")
                self._tiers[layer] = "aggressive"
            elif layer >= late_cut:
                self._strategies[layer] = _make_strategy("conservative")
                self._tiers[layer] = "conservative"
            else:
                self._strategies[layer] = _make_strategy("balanced")
                self._tiers[layer] = "balanced"

    def get_strategy(self, layer_idx: int):
        return self._strategies.get(layer_idx, _make_strategy("balanced"))

    def describe(self) -> Dict[int, str]:
        return dict(self._tiers)


class PerLayerIntervalSelector:
    """
    Custom anchor interval per-layer.
    Accepts a list of intervals, one per layer.
    """

    def __init__(self, intervals: List[int]):
        self._strategies = {i: PeriodicAnchorStrategy(interval=iv)
                             for i, iv in enumerate(intervals)}

    @classmethod
    def from_schedule(cls, num_layers: int, early_interval: int = 128,
                      mid_interval: int = 64, late_interval: int = 16,
                      early_frac: float = 0.25, late_frac: float = 0.25):
        """Build from early/mid/late schedule."""
        early_cut = int(num_layers * early_frac)
        late_cut  = int(num_layers * (1.0 - late_frac))
        intervals = []
        for i in range(num_layers):
            if i < early_cut:
                intervals.append(early_interval)
            elif i >= late_cut:
                intervals.append(late_interval)
            else:
                intervals.append(mid_interval)
        return cls(intervals)

    def get_strategy(self, layer_idx: int):
        return self._strategies.get(layer_idx, PeriodicAnchorStrategy(interval=64))

    def describe(self) -> Dict[int, str]:
        return {li: f"periodic_{s.interval}" for li, s in self._strategies.items()}


class PerLayerAdaptiveSelector:
    """
    Each layer gets its own independent adaptive policy.
    Policy type is chosen based on layer position:
    early → DynamicThreshold(adaptive), late → LayerNormalized
    """

    def __init__(self, num_layers: int, early_frac: float = 0.30,
                 late_frac: float = 0.30):
        self._strategies = {}
        early_cut = int(num_layers * early_frac)
        late_cut  = int(num_layers * (1.0 - late_frac))
        for layer in range(num_layers):
            if layer < early_cut:
                # Very smooth — use loose adaptive threshold targeting 2% anchor rate
                self._strategies[layer] = _make_adaptive("adaptive", target_rate=0.02)
            elif layer >= late_cut:
                # Rougher — use LayerNorm-based, tighter
                self._strategies[layer] = LayerNormalizedPolicy(
                    threshold=0.3, max_interval=128, min_interval=8
                )
            else:
                # Middle — EMA balanced
                self._strategies[layer] = EMAPolicy(
                    alpha=0.1, sensitivity_factor=2.5,
                    max_interval=256, min_interval=8
                )

    def get_strategy(self, layer_idx: int):
        return self._strategies.get(layer_idx, _make_strategy("balanced"))

    def describe(self) -> Dict[int, str]:
        return {li: type(s).__name__ for li, s in self._strategies.items()}


class LowRankScheduleSelector:
    """
    Assigns different ranks to different layers.
    Example: Strategy A (rank-4 early, rank-8 middle, FP16/INT8 late)
    """

    def __init__(self, num_layers: int, rank_schedule: List[Union[int, str]]):
        self.num_layers = num_layers
        self.rank_schedule = rank_schedule
        # If rank is 'dense' or 'fp16', we use a dense periodic strategy
        # If it's an int, it's the rank for low-rank compression
        self._strategies = {}
        for i, r in enumerate(rank_schedule):
            if isinstance(r, str) and (r == "fp16" or r == "dense"):
                self._strategies[i] = PeriodicAnchorStrategy(interval=16)
            elif isinstance(r, str) and r == "int8":
                self._strategies[i] = "int8_dkv" # Placeholder for int8 delta mode
            else:
                # Store the rank itself, we'll handle low-rank compression in the evaluator
                self._strategies[i] = f"lowrank_{r}"

    def get_strategy(self, layer_idx: int):
        return self._strategies.get(layer_idx, "lowrank_8")

    def describe(self) -> Dict[int, str]:
        return {li: str(s) for li, s in self._strategies.items()}


# ── Factory ────────────────────────────────────────────────────────────────────

def make_selector(mode: str, num_layers: int, **kwargs):
    """
    Factory for layer selectors.

    Modes:
      'early_only'   — EarlyOnlySelector
      'progressive'  — ProgressiveSelector
      'interval'     — PerLayerIntervalSelector
      'per_adaptive' — PerLayerAdaptiveSelector
      'uniform'      — all layers same strategy (pass strategy=...)
    """
    if mode == "early_only":
        return EarlyOnlySelector(num_layers, **kwargs)
    elif mode == "progressive":
        return ProgressiveSelector(num_layers, **kwargs)
    elif mode == "interval":
        return PerLayerIntervalSelector.from_schedule(num_layers, **kwargs)
    elif mode == "per_adaptive":
        return PerLayerAdaptiveSelector(num_layers, **kwargs)
    elif mode == "uniform":
        strategy_name = kwargs.get("strategy", "balanced")
        strat = _make_strategy(strategy_name)
        class UniformSelector:
            def get_strategy(self, layer_idx): return strat
            def describe(self): return {i: strategy_name for i in range(num_layers)}
        return UniformSelector()
    elif mode == "lowrank_schedule":
        return LowRankScheduleSelector(num_layers, **kwargs)
    else:
        raise ValueError(f"Unknown selector mode: {mode}")

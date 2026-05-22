"""
anchor_logic/strategies.py

Anchor placement strategies for Differential KV Cache.
Two strategies are provided:
  - PeriodicAnchorStrategy: inserts anchors at fixed token intervals.
  - AdaptiveAnchorStrategy: inserts anchors when delta magnitude or
    estimated reconstruction error exceeds configurable thresholds.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Union


class AnchorStrategy:
    """Base class for all anchor strategies."""
    def should_anchor(self, token_idx: int, kv: torch.Tensor,
                      last_anchor_kv: torch.Tensor, last_anchor_idx: int) -> 'AnchorDecision':
        raise NotImplementedError


@dataclass
class AnchorDecision:
    """Result of evaluating whether a token position needs a new anchor."""
    is_anchor: bool
    reason: str  # "periodic", "adaptive_magnitude", "adaptive_error", "forced"
    delta_norm: float = 0.0
    reconstruction_error_estimate: float = 0.0


class PeriodicAnchorStrategy(AnchorStrategy):
    """
    Insert a new anchor every `interval` tokens.

    Parameters
    ----------
    interval : int
        Token interval between anchors. E.g. 64 means every 64th token
        is guaranteed to be an anchor.
    """

    def __init__(self, interval: int = 64):
        assert interval > 0, "Anchor interval must be positive."
        self.interval = interval

    def should_anchor(self, token_idx: int, kv: torch.Tensor,
                      last_anchor_kv: torch.Tensor, last_anchor_idx: int) -> AnchorDecision:
        """
        Parameters
        ----------
        token_idx       : absolute token position
        kv              : KV tensor at this position, shape [2, heads, head_dim]
        last_anchor_kv  : KV tensor at the last anchor
        last_anchor_idx : token index of the last anchor

        Returns
        -------
        AnchorDecision
        """
        tokens_since_anchor = token_idx - last_anchor_idx
        if tokens_since_anchor >= self.interval:
            return AnchorDecision(is_anchor=True, reason="periodic",
                                  delta_norm=0.0)
        return AnchorDecision(is_anchor=False, reason="none")

    def __repr__(self):
        return f"PeriodicAnchorStrategy(interval={self.interval})"


class AdaptiveAnchorStrategy(AnchorStrategy):
    """
    Insert anchors both periodically AND whenever the delta magnitude or
    estimated reconstruction error exceeds a threshold.

    This prevents long, high-error delta chains from accumulating.

    Parameters
    ----------
    max_interval : int
        Hard maximum tokens between anchors regardless of delta quality.
    delta_norm_threshold : float
        If L2 norm of (kv - last_anchor_kv) exceeds this, force a new anchor.
    error_estimate_threshold : float
        If estimated reconstruction error exceeds this, force a new anchor.
        Currently estimated as: norm(kv - last_anchor_kv) / norm(last_anchor_kv + 1e-6)
    min_interval : int
        Minimum tokens between anchors (prevents anchor spam).
    """

    def __init__(
        self,
        max_interval: int = 64,
        delta_norm_threshold: float = 2.0,
        error_estimate_threshold: float = 0.05,
        min_interval: int = 8,
    ):
        self.max_interval = max_interval
        self.delta_norm_threshold = delta_norm_threshold
        self.error_estimate_threshold = error_estimate_threshold
        self.min_interval = min_interval

    def should_anchor(self, token_idx: int, kv: torch.Tensor,
                      last_anchor_kv: torch.Tensor, last_anchor_idx: int) -> AnchorDecision:
        tokens_since_anchor = token_idx - last_anchor_idx

        # Never anchor too soon
        if tokens_since_anchor < self.min_interval:
            return AnchorDecision(is_anchor=False, reason="none")

        # Periodic hard limit
        if tokens_since_anchor >= self.max_interval:
            return AnchorDecision(is_anchor=True, reason="periodic")

        # Compute delta statistics
        delta = kv.float() - last_anchor_kv.float()
        delta_norm = delta.norm().item()

        # Relative error estimate
        anchor_norm = last_anchor_kv.float().norm().item()
        rel_error = delta_norm / (anchor_norm + 1e-6)

        if delta_norm > self.delta_norm_threshold:
            return AnchorDecision(
                is_anchor=True,
                reason="adaptive_magnitude",
                delta_norm=delta_norm,
                reconstruction_error_estimate=rel_error,
            )

        if rel_error > self.error_estimate_threshold:
            return AnchorDecision(
                is_anchor=True,
                reason="adaptive_error",
                delta_norm=delta_norm,
                reconstruction_error_estimate=rel_error,
            )

        return AnchorDecision(
            is_anchor=False,
            reason="none",
            delta_norm=delta_norm,
            reconstruction_error_estimate=rel_error,
        )

    def __repr__(self):
        return (
            f"AdaptiveAnchorStrategy("
            f"max_interval={self.max_interval}, "
            f"delta_norm_thresh={self.delta_norm_threshold}, "
            f"error_thresh={self.error_estimate_threshold})"
        )

"""
anchor_logic/anchor_manager.py (Updated for Phase 2)

Upgrades the AnchorManager to support the full suite of Phase 2
adaptive policies and dynamic threshold tracking.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from dataclasses import dataclass, field

from .strategies import AnchorStrategy, AnchorDecision, PeriodicAnchorStrategy
from .adaptive_policies import AdaptivePolicyBase, AbsoluteNormalizedPolicy


@dataclass
class CompressionStats:
    num_tokens: int
    num_anchors: int
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    anchor_density: float
    anchor_reasons: Dict[str, int] = field(default_factory=dict)


class AnchorManager:
    """
    Manages anchor placement and delta-chain bookkeeping.
    Updated in Phase 2 to support smarter adaptive policies.

    Parameters
    ----------
    strategy : AnchorStrategy or AdaptivePolicyBase
    """

    def __init__(self, strategy: Union[AnchorStrategy, AdaptivePolicyBase] = None):
        self.strategy = strategy or PeriodicAnchorStrategy(interval=64)
        self.reset()

    def reset(self):
        self.anchors: Dict[int, torch.Tensor] = {}
        self.deltas:  Dict[int, torch.Tensor] = {}
        self.index_list: List[int] = []  # sorted list of anchor indices
        self.anchor_reasons: Dict[str, int] = {}
        self._last_stats: Optional[CompressionStats] = None

    def is_anchor(self, token_idx: int) -> bool:
        return token_idx in self.anchors

    def get_preceding_anchor(self, token_idx: int) -> Tuple[int, torch.Tensor]:
        """Find the nearest anchor <= token_idx using binary search."""
        import bisect
        if not self.index_list:
            raise RuntimeError("No anchors available. Call compress() first.")

        # Find the insertion point for token_idx
        idx = bisect.bisect_right(self.index_list, token_idx) - 1
        anchor_idx = self.index_list[idx]
        return anchor_idx, self.anchors[anchor_idx]

    def compress(self, kv_sequence: torch.Tensor) -> CompressionStats:
        """
        Compress an entire KV sequence [seq_len, 2, heads, dim].
        Applies the selected strategy/policy to place anchors.
        """
        from compression.quantization import quantize_int8

        self.reset()
        seq_len = kv_sequence.shape[0]

        last_anchor_idx = 0
        last_anchor_kv  = kv_sequence[0]
        self.anchors[0] = last_anchor_kv
        self.index_list.append(0)
        self.anchor_reasons["initial"] = 1

        total_comp_bytes = last_anchor_kv.numel() * 2  # FP16 anchor

        for i in range(1, seq_len):
            kv = kv_sequence[i]

            # Ask strategy for decision
            # Support both Phase 1 and Phase 2 policy interfaces
            if hasattr(self.strategy, "should_anchor"):
                decision = self.strategy.should_anchor(i, kv, last_anchor_kv, last_anchor_idx)
            else:
                # Fallback for legacy Periodic strategy
                interval = getattr(self.strategy, "interval", 64)
                if i - last_anchor_idx >= interval:
                    decision = AnchorDecision(is_anchor=True, reason="periodic")
                else:
                    decision = AnchorDecision(is_anchor=False, reason="none")

            if decision.is_anchor:
                self.anchors[i] = kv
                self.index_list.append(i)
                last_anchor_idx = i
                last_anchor_kv  = kv
                self.anchor_reasons[decision.reason] = self.anchor_reasons.get(decision.reason, 0) + 1
                total_comp_bytes += kv.numel() * 2
            else:
                # Store as delta relative to last anchor
                delta = kv.float() - last_anchor_kv.float()
                q_delta = quantize_int8(delta)
                self.deltas[i] = q_delta
                # INT8 + FP32 scale (approx 1.1 bytes per element)
                total_comp_bytes += q_delta.nbytes()

        # Calculate stats
        orig_bytes = seq_len * kv_sequence[0].numel() * 2
        stats = CompressionStats(
            num_tokens=seq_len,
            num_anchors=len(self.anchors),
            total_original_bytes=orig_bytes,
            total_compressed_bytes=total_comp_bytes,
            compression_ratio=orig_bytes / (total_comp_bytes + 1e-9),
            anchor_density=len(self.anchors) / seq_len,
            anchor_reasons=dict(self.anchor_reasons)
        )
        self._last_stats = stats
        return stats

    def get_delta(self, token_idx: int) -> Optional[torch.Tensor]:
        return self.deltas.get(token_idx)
